"""Pre-specified seed-level statistics, integrity checks, and training report."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from archive.experiments import quantizer_study as old
from archive.experiments.full_quantizer_training import SPECS, STAGES, FINGERPRINT, key, wire_bits, tensor_sha
from archive.compression.quantizer_payload import METHODS, canonical, encode, decode, CodebookRegistry

A, B, C, D, E, F = METHODS
PRIMARY = [(rep, bits, new, ref) for rep in ('SRK', 'Kashin')
           for bits, new, ref in ((1, E, C), (2, E, C), (2, F, D))]


def paired_summary(differences):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else None
    sd = float(values.std(ddof=1)) if n > 1 else None
    margin = float(t.ppf(.975, n-1)*sd/np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean_pp=mean, sample_sd_pp=sd,
                ci_low_pp=mean-margin if margin is not None else None,
                ci_high_pp=mean+margin if margin is not None else None)


def target(rows, threshold, sustained=False):
    ordered = sorted(rows, key=lambda r: r['round'])
    streak = 0
    previous = 0
    for row in ordered:
        if row.get('status', 'complete') != 'complete':
            return dict(status=row['status'], round=None, uplink_bits=None)
        if row['round'] != previous+1:
            return dict(status='execution_failed', round=None, uplink_bits=None)
        previous = row['round']
        streak = streak+1 if row['accuracy'] >= threshold else 0
        if streak >= (3 if sustained else 1):
            return dict(status='reached', round=row['round'], uplink_bits=row['total_uplink_bits'])
    return dict(status='not_reached' if len(ordered) == 50 else 'execution_failed', round=None, uplink_bits=None)


def endpoint(rows):
    by_round = {r['round']: r for r in rows}
    if len(by_round) != len(rows):
        raise ValueError('Duplicate endpoint rounds')
    failed = next((r for r in rows if r.get('status', 'complete') != 'complete'), None)
    if failed:
        return dict(status=failed['status'], accuracy=None, tail_accuracy=None, test_loss=None)
    if set(by_round) != set(range(1, 51)):
        return dict(status='execution_failed', accuracy=None, tail_accuracy=None, test_loss=None)
    return dict(status='complete', accuracy=by_round[50]['accuracy'],
                tail_accuracy=float(np.mean([by_round[r]['accuracy'] for r in range(46, 51)])),
                test_loss=by_round[50]['test_loss'])


def validate(rows, seeds, rounds):
    expected = {(s, key(spec), r) for s in seeds for spec in SPECS for r in range(1, rounds+1)}
    found = set()
    for row in rows:
        item = (row['seed'], row['canonical_config'], row['round'])
        if item in found:
            raise ValueError(f'Duplicate row: {item}')
        found.add(item)
        spec = (row['representation'], row['bits'], row['method'])
        if spec not in SPECS or key(spec) != row['canonical_config']:
            raise ValueError('Alias/noncanonical configuration or mismatched key')
        aliases = [m for m in METHODS if m != spec[2] and canonical(m, spec[1]) == spec[2]] if spec[1] else []
        if row['aliases'] != aliases:
            raise ValueError('Alias labels mismatch')
        if row['status'] == 'complete':
            if row['total_uplink_bits'] != wire_bits(spec)*row['round'] or row['round_uplink_bits'] != wire_bits(spec):
                raise ValueError('Incorrect actual payload accounting')
            if not (0 <= row['accuracy'] <= 1) or not np.isfinite(row['test_loss']):
                raise ValueError('Non-finite or wrong accuracy units')
            fields = ('local_training_seconds', 'compression_seconds', 'transform_seconds', 'encode_seconds',
                      'decode_seconds', 'inverse_seconds', 'aggregation_seconds', 'server_apply_seconds',
                      'evaluation_seconds', 'diagnostic_seconds', 'serial_algorithm_seconds', 'snapshot_io_seconds')
            if any(f not in row or row[f] < 0 for f in fields):
                raise ValueError('Missing/invalid timing fields')
        elif row['status'] not in ('diverged', 'execution_failed'):
            raise ValueError('Unknown failure status')
    if found != expected:
        raise ValueError(f'Missing/unexpected rows: missing={len(expected-found)} extra={len(found-expected)}')
    # Actual clients must be paired across every unique configuration per seed/round.
    for s in seeds:
        for rnd in range(1, rounds+1):
            clients = {tuple(r['clients']) for r in rows if r['seed'] == s and r['round'] == rnd}
            if len(clients) != 1 or len(next(iter(clients))) != 3:
                raise ValueError('Client pairing mismatch')
    return dict(unique_rows=len(found), expected_rows=len(expected), seeds=list(seeds), rounds=rounds,
                failed_rows=sum(r['status'] != 'complete' for r in rows), alias_rows=0)


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(endpoints, new_key, ref_key, label):
    index = {(r['seed'], r['canonical_config']): r for r in endpoints}
    differences, states = [], []
    for seed in range(5):
        new = index.get((seed, new_key), dict(status='execution_failed'))
        ref = index.get((seed, ref_key), dict(status='execution_failed'))
        ok = new['status'] == ref['status'] == 'complete'
        delta = 100*(new['accuracy']-ref['accuracy']) if ok else None
        if ok:
            differences.append(delta)
        states.append(dict(seed=seed, new_status=new['status'], reference_status=ref['status'], difference_pp=delta,
                           tail_difference_pp=100*(new['tail_accuracy']-ref['tail_accuracy']) if ok else None,
                           ce_difference=new['test_loss']-ref['test_loss'] if ok else None))
    return dict(comparison=label, new=new_key, reference=ref_key, **paired_summary(differences),
                per_seed=states, **{f'seed_{r["seed"]}_pp': r['difference_pp'] for r in states})


def stage_evaluate(root):
    """Same Original inputs; post-training diagnostics never touch training RNG/model."""
    destination = root/'stage_evaluation'
    destination.mkdir(exist_ok=False)
    registry = CodebookRegistry.load(root/'assets/codebooks.json')
    raw = []
    for seed in range(5):
        folder = root/f'seed_{seed}'
        manifest = json.loads((folder/'stage_inputs/manifest.json').read_text(encoding='utf8'))
        frame, signs = old.transforms(seed, 50890)
        if {s['round'] for s in manifest} != set(STAGES):
            raise ValueError('Missing Original stage snapshot')
        for record in manifest:
            path = folder/record['file']
            if old.sha(path) != record['file_sha256']:
                raise ValueError('Stage file hash mismatch')
            xs = torch.load(path, weights_only=True)['updates']
            if tensor_sha(xs) != record['tensor_sha256']:
                raise ValueError('Stage tensor hash mismatch')
            for rep in ('SRK', 'Kashin'):
                aa = old.coefficients(xs, rep, frame, signs)
                for bits, methods in ((1, (C, E)), (2, (C, E, D, F))):
                    cache = {}
                    for method in methods:
                        recs, cmse, cnmse = [], [], []
                        for a, x in zip(aa, xs):
                            p = encode(a, bits, method, rep, registry)
                            q = decode(p, registry)
                            rec = old.inverse(q, rep, frame, signs, 50890)
                            recs.append(rec)
                            cmse.append(((q.double()-a.double()).square().sum()/a.double().square().sum().clamp_min(1e-300)).item())
                            cnmse.append(((rec.double()-x.double()).square().sum()/x.double().square().sum().clamp_min(1e-300)).item())
                        # Match training's float32 aggregate before double error calculation.
                        nmse = ((torch.stack(recs).mean(0).double()-xs.double().mean(0)).square().sum()/xs.double().square().sum(1).mean().clamp_min(1e-300)).item()
                        row = dict(seed=seed, round=record['round'], representation=rep, bits=bits, method=method,
                                   normalized_coefficient_mse=float(np.mean(cmse)), client_nmse=float(np.mean(cnmse)), aggregate_nmse=nmse)
                        cache[method] = row
                        raw.append(row)
                    for new, ref in ((E, C),) if bits == 1 else ((E, C), (F, D)):
                        cache[new].update(reference=ref, **{f'{m}_fixed_fitted_ratio': cache[new][m]/max(cache[ref][m], 1e-300)
                                              for m in ('normalized_coefficient_mse', 'client_nmse', 'aggregate_nmse')})
            print(f'stage seed={seed} round={record["round"]}', flush=True)
    csv_write(destination/'same_input_raw.csv', raw)
    summary = []
    metrics = ('normalized_coefficient_mse', 'client_nmse', 'aggregate_nmse')
    for rnd in STAGES:
        for rep in ('SRK', 'Kashin'):
            for bits, methods in ((1, (C, E)), (2, (C, E, D, F))):
                for method in methods:
                    rr = [r for r in raw if (r['round'], r['representation'], r['bits'], r['method']) == (rnd, rep, bits, method)]
                    summary.append(dict(round=rnd, representation=rep, bits=bits, method=method, n=len(rr),
                        **{m:float(np.mean([r[m] for r in rr])) for m in metrics},
                        **{m:float(np.mean([r[m] for r in rr])) for m in rr[0] if m.endswith('_ratio')}))
    csv_write(destination/'same_input_summary.csv', summary)
    assert registry.fingerprint() == FINGERPRINT
    return summary


def make_figures(root, rows, comparisons, timing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.plot_style import configure_chinese_plotting
    configure_chinese_plotting()
    out = root/'analysis/figures'
    out.mkdir()
    labels = dict(zip(METHODS, ['A 随机 min/max', 'B 最近邻 min/max', 'C 拟合等距', 'D Lloyd-Max', 'E 固定等距', 'F 固定 Lloyd-Max']))
    for rep in ('SRK', 'Kashin'):
        for bits in (1, 2):
            specs = [('Original', 0, 'none')]+[s for s in SPECS if s[:2] == (rep, bits)]
            for horizontal in ('round', 'total_uplink_bits'):
                fig, ax = plt.subplots(figsize=(9, 5))
                for spec in specs:
                    rr = [r for r in rows if r['canonical_config'] == key(spec)]
                    means, sd, xx = [], [], []
                    for rnd in range(1, 51):
                        points = [r for r in rr if r['round'] == rnd and r['status'] == 'complete']
                        means.append(np.mean([r['accuracy'] for r in points])*100 if points else np.nan)
                        sd.append(np.std([r['accuracy'] for r in points], ddof=1)*100 if len(points)>1 else np.nan)
                        xx.append(rnd if horizontal == 'round' else wire_bits(spec)*rnd/1e6)
                    line, = ax.plot(xx, means, label='Original 原始更新' if spec[0] == 'Original' else labels[spec[2]],
                                    color='black' if spec[0] == 'Original' else None)
                    ax.fill_between(xx, np.array(means)-sd, np.array(means)+sd, alpha=.10, color=line.get_color())
                ax.set(ylim=(0, 100), xlabel='训练轮次' if horizontal == 'round' else '累计理想上行（Mbit，对数轴）',
                       ylabel='测试准确率（%）', title=f'{rep} / {bits}-bit：均值，阴影为跨 seed 标准差')
                if horizontal != 'round':
                    ax.set_xscale('log')
                ax.grid(alpha=.2)
                ax.legend(fontsize=9)
                fig.tight_layout()
                fig.savefig(out/f'{rep}_{bits}_{horizontal}.png', dpi=160)
                plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            times = [r for r in timing if (r['representation'], r['bits']) == (rep, bits)]
            for ax, metric, title in zip(axes, ('encode_seconds', 'compression_seconds'), ('累计在线编码', '累计完整压缩')):
                values = [np.mean([r[metric] for r in times if r['method'] == s[2]]) for s in specs[1:]]
                ax.bar([labels[s[2]].split(' ')[0] for s in specs[1:]], values)
                ax.set(ylabel='秒 / 50轮（5 seed 均值）', title=title)
            fig.suptitle(f'{rep} / {bits}-bit；串行模拟，不含网络')
            fig.tight_layout()
            fig.savefig(out/f'{rep}_{bits}_time.png', dpi=160)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(comparisons):
        if row['n']:
            mean = row['mean_pp']
            ax.errorbar(mean, i, xerr=[[mean-row['ci_low_pp']], [row['ci_high_pp']-mean]] if row['n']>1 else None, fmt='o')
            ax.scatter([s['difference_pp'] for s in row['per_seed'] if s['difference_pp'] is not None], [i]*row['n'], alpha=.4)
    ax.axvline(0, color='black', linestyle='--')
    ax.set(yticks=range(len(comparisons)), yticklabels=[r['comparison'] for r in comparisons],
           xlabel='第50轮准确率差（百分点，新方案−对照）', title='六项预定比较：seed 配对差；未校正近似95% t区间')
    fig.tight_layout()
    fig.savefig(out/'primary_paired.png', dpi=160)
    plt.close(fig)


def summarize(root):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text(encoding='utf8'))
    if len(manifest['formal_runs']) != 5 or any(r['status'] != 'complete' for r in manifest['formal_runs']):
        raise ValueError('Formal manifest incomplete/failed; cannot claim five-seed completion')
    rows, payloads, metas, io = [], [], [], []
    for entry in manifest['formal_runs']:
        folder = root/entry['directory']
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        if meta['status'] != 'complete' or meta['binding'] != manifest['binding'] or meta['final_binding'] != manifest['binding']:
            raise ValueError('Formal binding/status mismatch')
        rows.extend(json.loads((folder/'rounds.json').read_text(encoding='utf8')))
        payloads.extend(json.loads((folder/'payloads.json').read_text(encoding='utf8')))
        metas.append(meta)
        io.extend(json.loads((folder/'io.json').read_text(encoding='utf8')))
    integrity = validate(rows, range(5), 50)
    analysis = root/'analysis'
    analysis.mkdir(exist_ok=False)
    old.write_json(analysis/'integrity.json', integrity)
    endpoints, targets, timing = [], [], []
    time_fields = ('local_training_seconds', 'transform_seconds', 'encode_seconds', 'decode_seconds', 'inverse_seconds',
                   'aggregation_seconds', 'compression_seconds', 'server_apply_seconds', 'serial_algorithm_seconds',
                   'evaluation_seconds', 'diagnostic_seconds', 'snapshot_io_seconds')
    for seed in range(5):
        for spec in SPECS:
            rr = [r for r in rows if (r['seed'], r['canonical_config']) == (seed, key(spec))]
            identity = dict(seed=seed, canonical_config=key(spec), representation=spec[0], bits=spec[1], method=spec[2])
            endpoints.append(dict(**identity, **endpoint(rr)))
            timing.append(dict(**identity, **{f:sum(r.get(f, 0.) for r in rr) for f in time_fields}))
            for threshold in (.85, .90, .92):
                for sustained in (False, True):
                    targets.append(dict(**identity, threshold=threshold, definition='sustained3' if sustained else 'first',
                                        **target(rr, threshold, sustained)))
    primary = [compare(endpoints, key((rep, b, new)), key((rep, b, ref)), f'{rep} {b}bit {"ABCDEF"[METHODS.index(new)]}−{"ABCDEF"[METHODS.index(ref)]}')
               for rep, b, new, ref in PRIMARY]
    secondary = []
    for rep in ('SRK', 'Kashin'):
        for b in (1, 2):
            for new, ref in [(B, A), (C, B), (E, A)]+([(D, C), (F, A)] if b == 2 else []):
                secondary.append(compare(endpoints, key((rep, b, new)), key((rep, b, ref)), f'{rep} {b}bit {new}−{ref}'))
    for spec in SPECS[1:]:
        secondary.append(compare(endpoints, key(spec), key(SPECS[0]), f'{key(spec)}−Original'))
        if spec[0] == 'Kashin':
            secondary.append(compare(endpoints, key(spec), key(('SRK', spec[1], spec[2])), f'Kashin−SRK {spec[1]}bit {spec[2]}'))
    for name, records in (('endpoints', endpoints), ('targets', targets), ('timing_summary', timing),
                           ('paired_primary', primary), ('paired_secondary', secondary)):
        csv_write(analysis/f'{name}.csv', [{k:v for k,v in r.items() if k != 'per_seed'} for r in records])
        old.write_json(analysis/f'{name}.json', records)
    # Common achiever paired results, retaining every seed's status explicitly.
    target_pairs = []
    target_index = {(r['seed'], r['canonical_config'], r['threshold'], r['definition']):r for r in targets}
    for comp in primary+secondary:
        for threshold in (.85, .9, .92):
            for definition in ('first', 'sustained3'):
                per_seed = []
                for seed in range(5):
                    new = target_index[(seed, comp['new'], threshold, definition)]
                    ref = target_index[(seed, comp['reference'], threshold, definition)]
                    ok = new['status'] == ref['status'] == 'reached'
                    per_seed.append(dict(seed=seed, new_status=new['status'], reference_status=ref['status'],
                                         uplink_difference_bits=new['uplink_bits']-ref['uplink_bits'] if ok else None,
                                         round_difference=new['round']-ref['round'] if ok else None))
                vals = [r['uplink_difference_bits'] for r in per_seed if r['uplink_difference_bits'] is not None]
                target_pairs.append(dict(comparison=comp['comparison'], threshold=threshold, definition=definition,
                                         common_achievers=len(vals), mean_uplink_difference_bits=float(np.mean(vals)) if vals else None,
                                         new_achievers=sum(r['new_status']=='reached' for r in per_seed),
                                         reference_achievers=sum(r['reference_status']=='reached' for r in per_seed), per_seed=per_seed))
    old.write_json(analysis/'target_pairs.json', target_pairs)
    csv_write(analysis/'target_pairs.csv', [{k:v for k,v in r.items() if k != 'per_seed'} for r in target_pairs])
    own = [r for r in payloads if 'normalized_coefficient_mse' in r]
    csv_write(analysis/'fixed_own_trajectory.csv', own)
    stages = stage_evaluate(root)
    make_figures(root, rows, primary, timing)
    report(root, integrity, endpoints, primary, secondary, timing, targets, target_pairs, stages, metas, io)


def report(root, integrity, endpoints, primary, secondary, timing, targets, target_pairs, stages, metas, io):
    lines = ['# 冻结量化器完整训练报告', '', '## 1. 实际完成与边界', '',
             f'本轮完成5个seed，每seed 21条唯一模型路径×50轮，共{integrity["unique_rows"]}条记录；异常状态行{integrity["failed_rows"]}条。不是10月4日两轮烟测结果。',
             '输入：MNIST前6000训练图、10000测试图；MLP 784→64→10，50890参数；10客户端每轮3个，2本地epochs、batch32、SGD学习率0.05；CPU Torch两线程。D=65536，λ=65536/50890。',
             f'码本内容指纹 `{FINGERPRINT}`。源文件、数据字节哈希和协议见根目录manifest；未提交源码完整快照随结果保留。',
             '统计条件于这一份旧冻结码本；seed0/1/2部分历史结果已见过，非盲测。没有重新校准，没有更换差结果路径。1bit C=D、E=F只是别名，不重复训练或计费。', '',
             '## 2. 六项主要配对结果', '',
             '表中是新方案−对照，单位为百分点。区间为5个seed配对差的未校正近似95% t区间；不是等效性/非劣检验。', '',
             '| 比较 | 各seed差值（百分点） | 均值 | 95%区间 | 编码减少 | 完整压缩减少 |',
             '|---|---|---:|---|---:|---:|']
    def average(config, field):
        return float(np.mean([r[field] for r in timing if r['canonical_config'] == config]))
    for r in primary:
        vals = ', '.join(f'{v["difference_pp"]:+.3f}' if v['difference_pp'] is not None else v['new_status'] for v in r['per_seed'])
        lo, hi = r['ci_low_pp'], r['ci_high_pp']
        encode_gain = 1-average(r['new'], 'encode_seconds')/average(r['reference'], 'encode_seconds')
        full_gain = 1-average(r['new'], 'compression_seconds')/average(r['reference'], 'compression_seconds')
        lines.append(f'| {r["new"]} − {r["reference"]} | {vals} | {r["mean_pp"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {encode_gain:.1%} | {full_gain:.1%} |')
    lines += ['', '均值接近不代表保证无损；区间包含的下降/提升幅度都需考虑。逐seed末五轮和交叉熵差在paired_primary.json；主终点固定第50轮，未取最高点。', '',
              '## 3. 全部终点与相对随机量化', '', '| 表示 | bit | 方法 | 第50轮准确率均值±seed标准差（%） | 末五轮均值（%） | 第50轮交叉熵 | 完整压缩秒/50轮 |', '|---|---:|---|---:|---:|---:|---:|']
    for spec in SPECS:
        rr = [r for r in endpoints if r['canonical_config'] == key(spec) and r['status'] == 'complete']
        if not rr:
            continue
        lines.append(f'| {spec[0]} | {spec[1]} | {spec[2]} | {np.mean([r["accuracy"] for r in rr])*100:.3f}±{np.std([r["accuracy"] for r in rr],ddof=1)*100:.3f} | {np.mean([r["tail_accuracy"] for r in rr])*100:.3f} | {np.mean([r["test_loss"] for r in rr]):.5f} | {average(key(spec), "compression_seconds"):.3f} |')
    lines += ['', 'E−A（及2bit F−A）的准确率差：', '']
    for r in secondary:
        if r['reference'].endswith(A) and (r['new'].endswith(E) or r['new'].endswith(F)):
            lines.append(f'- {r["comparison"]}：{r["mean_pp"]:+.3f}个百分点，95%区间[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]。')
    lines += ['', '这是真实训练路径的结果，不能用低NMSE替代准确率结论。全部次要配对（含B−A、C−B、2bit D−C、对Original）均在paired_secondary中，不是只挑有利项。', '',
              '## 4. Kashin相对SRK', '']
    for r in secondary:
        if r['comparison'].startswith('Kashin−SRK'):
            extra = average(r['new'], 'transform_seconds')-average(r['reference'], 'transform_seconds')
            lines.append(f'- {r["comparison"]}：准确率差{r["mean_pp"]:+.3f}个百分点，区间[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]；50轮额外表示时间{extra:.3f}秒。')
    lines += ['', '必须把准确率差与额外求解时间一起看，不能仅凭压缩失真优势断言训练更好，也不能把单一MNIST结论扩展到所有任务。', '',
              '## 5. 达标通信', '', 'first是首次达标；sustained3计费到连续3轮达标的第3轮。未达标不填成51轮或0bit。targets.csv保留所有seed状态，target_pairs.json只在共同达标seed上计算差值并列明n。', '',
              '| 配置 | 阈值 | 首次达标数/5 | 连续3轮达标数/5 |', '|---|---:|---:|---:|']
    for spec in SPECS:
        for threshold in (.85, .9, .92):
            counts = [sum(r['status']=='reached' for r in targets if r['canonical_config']==key(spec) and r['threshold']==threshold and r['definition']==definition) for definition in ('first', 'sustained3')]
            lines.append(f'| {key(spec)} | {threshold:.0%} | {counts[0]} | {counts[1]} |')
    lines += ['', '## 6. 中后期固定码本适配', '',
              '同输入：每seed Original正常训练第1/15/30/50轮聚合前3客户端更新。离线比较相同更新、相同表示的C/E与2bit D/F；没有回写模型。先取seed内3客户端均值，再跨5seed等权平均。表中ratio是先seed内fixed/fitted再取跨seed均值。', '',
              '| 表示 | bit | 固定方法 | 轮次 | 归一化系数MSE | 客户端NMSE | 聚合NMSE | 聚合fixed/fitted比 |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in stages:
        if r['method'].startswith('fixed-'):
            lines.append(f'| {r["representation"]} | {r["bits"]} | {r["method"]} | {r["round"]} | {r["normalized_coefficient_mse"]:.5g} | {r["client_nmse"]:.5g} | {r["aggregate_nmse"]:.5g} | {r["aggregate_nmse_fixed_fitted_ratio"]:.4f} |')
    lines += ['', 'E/F自身轨迹：fixed_own_trajectory.csv另存各客户端RMS、z的1%/50%/99%分位数、极端等级外比例、系数与实际更新重建误差。它与Original参考输入不同，不能混为同输入实验。超出极端重建等级不是丢包、索引错误，也不能单凭该比例判断失效。', '',
              '## 7. 时间与通信口径', '',
              '计时为一次真实执行，不重复拟合取最好时间。完整压缩直接计时：表示→encode→独立decode→逆变换→stack/mean；包含少量Python调度开销。各分项互斥；NMSE/分位数/日志转换在diagnostic_seconds，评估在evaluation_seconds，模型应用在server_apply_seconds。',
              '旧A/C/D编码助手内部会构造临时重建值，这是encode内原有工作；实际接收侧decode仍独立测量，没有把临时重建当额外decode再次加总。串行算法时间=本地训练+完整压缩+服务器应用；Original另加其实际聚合。',
              f'五个seed进程总墙钟{sum(r["process_wall_seconds"] for r in metas):.1f}秒，这是交替运行所有21路径的总和，不是某方法训练时间。初始化共{sum(r["initialization_seconds"] for r in metas):.3f}秒；共享逐轮JSON写盘共{sum(r["shared_round_log_io_seconds"] for r in io):.3f}秒（io.json本身最后一次写入未包含在该分项，包含在进程墙钟）。',
              '每行snapshot_io_seconds是该路径阶段快照IO；共享逐轮日志IO不能任意分摊成方法加速。逐轮源码/冻结核验等管理开销只在进程墙钟，不算算法时间。没有真实网络时延，不是分布式端到端加速。',
              '每客户端：A/B/C=D*b+64；D=D*b+32*2^b；E/F=D*b+32；Original=50890*32。每轮3客户端，actual payload元数据计入，alias不重复计费。同bit跨量化器预算相近但不完全相同；同量化器SRK/Kashin严格匹配。',
              '固定码本预装假设：20个float32中心净荷640bit；向10客户端分发整库下界6400bit，另有封装与配置。本轮上行不包括该下行分发，也不声称它是零成本。实际索引张量内存/JSON大小不是理想通信负载。',
              '准确率−上行图用实际轮次点的连线，x为对数轴以容纳Original；没有任意预算插值或外推。准确率图0–100%，阴影是跨seed标准差。', '',
              '## 8. 判断与下一步', '',
              '先以固定码本和逐向量拟合的实际配对区间判断准确率代价，再看在线编码节省与完整压缩节省。若Kashin收益很小/方向不稳定而求解更贵，应保留SRK强量化作为基线，不能只与较弱min/max比较。',
              '中后期误差比需结合自身轨迹和训练结果；即使增加，也只能作为下一轮针对性验证的理由，本轮没有更新码本。五seed、单一数据/网络、固定校准集仍不足以证明普遍等效、非劣或论文级创新。', '',
              '## 9. 可追溯文件', '',
              'protocol.json锁定设置与端点；source_manifest.json/source_snapshot保存未提交源码；run_manifest仅纳入5个正式seed；seed_*/rounds.json为原始逐轮结果；payloads.json为实际编码元数据；stage_inputs/含原始更新与模型哈希。analysis内CSV/JSON和figures对应以上表格。']
    (root/'analysis/Full_Training_Report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    (root/'analysis/Reproduce_Commands.md').write_text(
        '# 本轮实际入口\n\n工作目录 `D:\\FL`，解释器 `D:\\Miniconda\\envs\\fl-mnist\\python.exe`。\n\n'
        '```powershell\n& D:\\Miniconda\\envs\\fl-mnist\\python.exe -m experiments.run_full_quantizer_training --out outputs/results/quantizer_full_NEW_RUN\n```\n\n'
        '该命令先测试、烟测、冻结，随后顺序运行5×50轮、阶段诊断与汇总。输出目录必须全新。已有本轮完整命令见根目录commands.json。'
        '没有逐轮恢复功能；技术失败保留失败目录，必须以新目录重跑整个seed并登记manifest，不能拼接两轮烟测。\n', encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    summarize(args.root)
