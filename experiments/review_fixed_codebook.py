"""Read-only review of the completed run; never starts training or rewrites it."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import quantizer_study as old
from experiments.full_quantizer_training import SPECS, FINGERPRINT, key, tensor_sha, verify_binding
from experiments.summarize_quantizer_training import PRIMARY, endpoint, compare, target, validate
from compression.quantizer_payload import METHODS, CodebookRegistry

A, B, C, D, E, F = METHODS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def hashes(root):
    return {str(p.relative_to(root)): old.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.source).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        raise ValueError('Review output must be separate from original results')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    before = hashes(root)
    binding = verify_binding(root)
    source = read(root/'source_manifest.json')
    assert all(old.sha(root/'source_snapshot'/p) == digest for p, digest in source['files'].items())
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == binding['head']
    protocol, manifest = read(root/'protocol.json'), read(root/'run_manifest.json')
    assert manifest['status'] == 'complete' and manifest['binding'] == binding
    assert len(manifest['formal_runs']) == 5
    assert [r['seed'] for r in manifest['formal_runs']] == list(range(5))
    assert protocol['seeds'] == list(range(5)) and protocol['rounds'] == 50
    assert (protocol['D'], protocol['parameters'], protocol['clients'], protocol['clients_per_round'],
            protocol['local_epochs'], protocol['batch_size'], protocol['learning_rate']) == (65536,50890,10,3,2,32,.05)
    assert all(old.sha(Path(path)) == digest for path,digest in protocol['dataset_hashes'].items())
    registry = CodebookRegistry.load(root/'assets/codebooks.json')
    assert registry.fingerprint() == FINGERPRINT
    for rep in ('SRK','Kashin'):
        centers = registry.centers(rep,2,E)
        assert torch.allclose(centers.diff(), centers.diff()[0].expand(3), atol=2e-7)
        assert not torch.allclose(registry.centers(rep,2,F).diff(), centers.diff(), atol=1e-3)
    rows, logs, snapshot_count = [], [], 0
    for entry in manifest['formal_runs']:
        folder = root/entry['directory']
        meta = read(folder/'metadata.json')
        assert entry['status'] == meta['status'] == 'complete'
        assert meta['binding'] == meta['final_binding'] == binding
        assert meta['rounds'] == 50 and not meta['smoke']
        torch.manual_seed(entry['seed'])
        model = old.MNIST_MLP()
        assert tensor_sha(old.flatten_state_dict(model.state_dict())[0]) == meta['initialization_sha256']
        rr = read(folder/'rounds.json')
        validate(rr, [entry['seed']], 50)
        rng = np.random.default_rng(entry['seed']+8700)
        for rnd in range(1,51):
            clients = rng.choice(10,3,replace=False).tolist()
            assert all(r['clients'] == clients for r in rr if r['round'] == rnd)
        payloads = read(folder/'payloads.json')
        assert len(payloads) == 20*50*3
        index = {(r['round'],r['canonical_config']):r for r in rr}
        for p in payloads:
            spec = next(s for s in SPECS if key(s) == p['canonical_config'])
            if spec[2].startswith('fixed-'):
                assert p['diagnostics']['fit'] == 'offline-frozen'
                assert len(p['metadata']) == 1 and p['metadata'][0] >= 0
            assert p['wire_bits']*3 == index[p['round'],p['canonical_config']]['round_uplink_bits']
        for record in read(folder/'stage_inputs/manifest.json'):
            path = folder/record['file']
            assert record['binding'] == binding and record['global_model_round'] == record['round']-1
            assert old.sha(path) == record['file_sha256']
            saved = torch.load(path,weights_only=True)
            assert tensor_sha(saved['updates']) == record['tensor_sha256']
            assert tensor_sha(saved['global_before']) == record['global_sha256']
            snapshot_count += 1
        rows.extend(rr)
        logs.extend(payloads)
    integrity = validate(rows, range(5),50)
    validate(read(root/'smoke/rounds.json'),[0],2)
    old.write_json(out/'integrity.json',integrity)
    endpoints, timing, targets = [], [], []
    time_fields = ('encode_seconds','decode_seconds','transform_seconds','inverse_seconds','compression_seconds',
                   'local_training_seconds','server_apply_seconds','serial_algorithm_seconds','evaluation_seconds')
    for seed in range(5):
        for spec in SPECS:
            rr = [r for r in rows if r['seed']==seed and r['canonical_config']==key(spec)]
            ident = dict(seed=seed,canonical_config=key(spec),representation=spec[0],bits=spec[1],method=spec[2])
            ep = dict(**ident,**endpoint(rr))
            assert ep['status']=='complete'
            endpoints.append(ep)
            timing.append(dict(**ident,**{f:sum(r[f] for r in rr) for f in time_fields}))
            for threshold in (.85,.9,.92):
                for sustained in (False,True):
                    targets.append(dict(**ident,threshold=threshold,definition='sustained3' if sustained else 'first',
                                        **target(rr,threshold,sustained)))
    comparisons=[]
    def add(new,ref,group,label):
        comparison=compare(endpoints,key(new),key(ref),label)
        comparison['group']=group
        comparison['post_hoc']=group=='F−E'
        differences=[]
        for seed in range(5):
            n=next(r for r in timing if r['seed']==seed and r['canonical_config']==key(new))
            r=next(r for r in timing if r['seed']==seed and r['canonical_config']==key(ref))
            differences.append(dict(seed=seed,**{f:n[f]-r[f] for f in time_fields}))
        comparison['timing_differences_seconds']=differences
        comparison['target_pairs']=[]
        for threshold in (.85,.9,.92):
            for definition in ('first','sustained3'):
                states=[]
                for seed in range(5):
                    n=next(r for r in targets if (r['seed'],r['canonical_config'],r['threshold'],r['definition'])==(seed,key(new),threshold,definition))
                    r=next(r for r in targets if (r['seed'],r['canonical_config'],r['threshold'],r['definition'])==(seed,key(ref),threshold,definition))
                    ok=n['status']==r['status']=='reached'
                    states.append(dict(seed=seed,new_status=n['status'],reference_status=r['status'],
                                       difference_bits=n['uplink_bits']-r['uplink_bits'] if ok else None,
                                       difference_rounds=n['round']-r['round'] if ok else None))
                valid=[r['difference_bits'] for r in states if r['difference_bits'] is not None]
                comparison['target_pairs'].append(dict(threshold=threshold,definition=definition,n=len(valid),
                    mean_difference_bits=float(np.mean(valid)) if valid else None, per_seed=states))
        comparisons.append(comparison)
    for rep,b,new,ref in PRIMARY:
        add((rep,b,new),(rep,b,ref),'固定−拟合',f'{rep} {b}bit 固定−拟合 {new}')
    for rep in ('SRK','Kashin'):
        for b in (1,2):
            for m in (E,) if b==1 else (E,F):
                add((rep,b,m),(rep,b,A),'固定−随机',f'{rep} {b}bit {m}−随机')
        add((rep,2,F),(rep,2,E),'F−E',f'{rep} 2bit 固定Lloyd-Max−固定等距')
    for spec in SPECS[1:]:
        add(spec,SPECS[0],'对Original',f'{key(spec)}−Original')
        if spec[0]=='Kashin':
            add(spec,('SRK',spec[1],spec[2]),'Kashin−SRK',f'{spec[1]}bit {spec[2]} Kashin−SRK')
    old.write_json(out/'paired_comparisons.json',comparisons)
    old.write_json(out/'training_comparison.json',[dict(e,**{f:next(r[f] for r in timing if (r['seed'],r['canonical_config'])==(e['seed'],e['canonical_config'])) for f in time_fields}) for e in endpoints])
    old.write_json(out/'targets.json',targets)
    # Existing statistics must agree exactly, not merely after rounding.
    assert endpoints == read(root/'analysis/endpoints.json')
    assert timing == [{k:r[k] for k in timing[0]} for r in read(root/'analysis/timing_summary.json')]
    figures=out/'figures';figures.mkdir()
    for path in (root/'analysis/figures').glob('*.png'):
        shutil.copy2(path,figures/path.name)
    report(out,endpoints,timing,comparisons,targets)
    logpath=out/'tests.log'
    completed=subprocess.run([sys.executable,'-X','utf8','-m','pytest','tests','-q'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8')
    logpath.write_text(completed.stdout,encoding='utf8')
    print(completed.stdout,flush=True)
    if completed.returncode:
        raise RuntimeError('Review tests failed')
    assert hashes(root)==before,'Original result files changed during review'
    old.write_json(out/'input_manifest.json',dict(source=str(root),binding=binding,files_sha256=before,unchanged_after_review=True))
    shutil.copy2(__file__,out/'review_fixed_codebook.py')
    shutil.copy2('tests/test_fixed_review.py',out/'test_fixed_review.py')
    old.write_json(out/'commands.json',dict(cwd=str(Path.cwd()),review=[sys.executable,'-X','utf8','-m','experiments.review_fixed_codebook',*sys.argv[1:]],
                   tests=[sys.executable,'-X','utf8','-m','pytest','tests','-q'],source_sha256=old.sha(__file__)))
    (out/'verification.md').write_text('# 这次检查了什么\n\n'
        f'5250条正式记录完整，0异常；5seed各21路径50轮；两轮烟测单独保存、未拼接。{len(logs)}条客户端payload记录、{snapshot_count}个阶段文件及张量/模型哈希通过检查。\n\n'
        '当前计算源码与保存的源码快照逐文件相同。HEAD、协议、旧码本指纹与文件字节哈希一致；本地数据哈希一致。协议设置与局部训练源码符合要求。初始化模型哈希、各轮客户端抽样序列复算一致。每轮各方法复用同一minibatch列表，minibatch随机规则来自绑定源码；没有逐minibatch历史日志，因此不声称逐条重放训练。\n\n'
        '固定码本调用链：本地更新→表示→RMS→固定中心最近邻→仅payload和共享码本decode→逆变换→mean→apply_update。检查了绑定快照full_quantizer_training.py的instrumented_aggregate和train_seed、quantizer_payload.py的encode/decode、quantizer_study.py的apply_update。E/F只发一个float32尺度；日志为offline-frozen。原始payload日志不保存全部索引，也没有保存所有最终模型，所以不能事后逐bit重放全部训练。\n\n'
        '新增6项零解码注入测试：把真正decode的输出置零，聚合更新为零，服务器模型不变；同一非零原始更新直接应用会改变模型。覆盖两种表示的1bit E、2bit E/F。这是当前同版本代码的链路测试，不冒充历史运行逐轮注入记录。\n\n'
        '本次测试输出见tests.log，不能只引用上次49项通过。正式端点和计时从rounds.json重新计算，与旧JSON统计逐值一致。固定2bit等距中心间距核验通过，Lloyd-Max为非等间距。原结果目录复核前后全部文件SHA256相同，没有修改旧报告或重跑训练。\n\n'
        '图中的轮次、上行、时间来自旧正式运行；复制原图以保留所有方法和Original。新补F−E是事后分析，不替换预设六比较。未修改算法/码本，未commit或push。\n',encoding='utf8')
    print(f'COMPLETE {out}',flush=True)


def report(out,endpoints,timing,comparisons,targets):
    def mean(config,field,source=timing):
        return float(np.mean([r[field] for r in source if r['canonical_config']==config]))
    lines=['# 固定码本训练对比', '',
        '这次没有重新训练。我检查了已经跑完的5组实验，把结果放到一起比较。每组都有21种配置，每种跑50轮。原数据和原报告没有改。', '',
        '## 1. 固定码本真的用来训练了吗？', '',
        '用了。客户端先训练，得到原始更新；再变换、量化、解码、变回模型更新。服务器加到模型上的是解码后的更新，不是原始更新。新做的测试把解码结果设成零，模型就不再变化。',
        '“固定”是标准化后的几个量化值不变。每轮还会根据更新大小乘一个RMS尺度，所以实际端点会跟着变。没有每轮重新拟合中心。1bit下固定等距和固定Lloyd-Max共用同一条训练路径，2bit才分开比较。', '',
        '## 2. 比每轮重新拟合怎么样？', '',
        '准确率均值很接近，在线编码确实省时。但不能说完全一样，也不能把编码省时当成整轮加速。下面时间都是每种方法跑50轮后，再对5组取平均。', '',
        '| 表示/bit/固定方法 | 准确率差（百分点） | 差值95%区间 | 编码：拟合→固定（秒） | 完整压缩：拟合→固定（秒） |',
        '|---|---:|---|---|---|']
    for r in comparisons:
        if r['group']=='固定−拟合':
            lines.append(f'| {r["new"]} | {r["mean_pp"]:+.3f} | [{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}] | {mean(r["reference"],"encode_seconds"):.3f}→{mean(r["new"],"encode_seconds"):.3f} | {mean(r["reference"],"compression_seconds"):.3f}→{mean(r["new"],"compression_seconds"):.3f} |')
    lines += ['', '这里的“百分点”就是92.8%减92.7%等于0.1个百分点。区间表示5组结果还有多大不确定性；这些区间没有做多重比较校正。区间包含0，不等于已经证明两种方法一样。', '',
        '## 3. 比最早的随机量化怎么样？', '',
        '没有稳定更好。1bit时，固定等距的最终准确率略低，达到92%也更晚。2bit也没有看到固定码本稳定超过随机量化。', '',
        '| 表示 | bit | 固定方法 | 随机准确率（%） | 固定准确率（%） | 准确率差（百分点） | 完整压缩：随机→固定（秒） |', '|---|---:|---|---:|---:|---:|---|']
    for r in comparisons:
        if r['group']=='固定−随机':
            spec=r['new'].split(':')
            lines.append(f'| {spec[0]} | {spec[1]} | {spec[2]} | {100*mean(r["reference"],"accuracy",endpoints):.3f} | {100*mean(r["new"],"accuracy",endpoints):.3f} | {r["mean_pp"]:+.3f} | {mean(r["reference"],"compression_seconds"):.3f}→{mean(r["new"],"compression_seconds"):.3f} |')
    lines += ['', '这说明：固定码本主要省掉的是“每轮重新拟合”的计算。与原随机量化相比，不能说它在准确率和总压缩时间上都更好。', '',
        '| 1bit方法 | 首次92%的平均轮数 | 连续3轮确认的平均轮数 | 首次92%的平均上行（Mbit） | 达标组数 |', '|---|---:|---:|---:|---:|']
    for rep in ('SRK','Kashin'):
        for method in (A,E):
            config=key((rep,1,method))
            first=[r for r in targets if r['canonical_config']==config and r['threshold']==.92 and r['definition']=='first']
            sustained=[r for r in targets if r['canonical_config']==config and r['threshold']==.92 and r['definition']=='sustained3']
            assert all(r['status']=='reached' for r in first+sustained)
            lines.append(f'| {rep} {"随机" if method==A else "固定等距"} | {np.mean([r["round"] for r in first]):.1f} | {np.mean([r["round"] for r in sustained]):.1f} | {np.mean([r["uplink_bits"] for r in first])/1e6:.3f} | 5/5 |')
    lines += ['', '85%、90%、92%的全部逐组状态保存在targets.json。没有达标就记未达标，不填0bit或51轮。', '',
        '## 4. 固定Lloyd-Max比固定等距更好吗？', '', '下面是2bit的直接比较，按相同seed逐组相减。这是看完原实验后补充的分析，不是原六项主要比较。', '',
        '| 表示 | 五组准确率差（百分点，Lloyd-Max−等距） | 平均差 | 95%区间 |', '|---|---|---:|---|']
    for r in comparisons:
        if r['group']=='F−E':
            ds=', '.join(f'{s["difference_pp"]:+.3f}' for s in r['per_seed'])
            lines.append(f'| {r["new"].split(":")[0]} | {ds} | {r["mean_pp"]:+.3f} | [{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}] |')
    lines += ['', '没有看到稳定的额外训练收益。量化值可以自由摆放，不代表最终准确率一定更高。1bit不能再做独立的Lloyd-Max对比，因为两者本来就共用结果。', '',
        '## 5. Kashin多花的计算值得吗？', '',
        '这次还没有足够证据说值得。给两者相同的量化器后，准确率差很小，方向也不一致，但Kashin求解明显更花时间。', '',
        '| bit/方法 | Kashin−SRK准确率差（百分点） | 95%区间 | 多花的表示求解时间（秒/50轮） |', '|---|---:|---|---:|']
    for r in comparisons:
        if r['group']=='Kashin−SRK':
            lines.append(f'| {r["new"].split(":",1)[1]} | {r["mean_pp"]:+.3f} | [{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}] | {mean(r["new"],"transform_seconds")-mean(r["reference"],"transform_seconds"):.3f} |')
    lines += ['', 'B是直接拿min/max做最近邻，它在1bit时两者都很差。Kashin在这个较弱对照上更好，不足以说明它超过了SRK的强量化方案。图和数据里没有删除B。', '',
        '## 6. 这次应该怎么理解？', '',
        '固定码本是一个省在线拟合计算的办法，目前不是一个提高准确率的办法。失真小、准确率高、训练快，是三件不同的事。',
        '下一步可以先检查“误差方向和偏差”，解释为什么失真变小却没有训得更好。这只是下一轮建议，这次没有改算法、加误差反馈或重新校准。',
        '结论只适用于这份MNIST、这个小MLP和这份码本。5组结果不多，部分seed以前看过，不能说完全盲测，也不能保证换任务后仍然一样。', '',
        '## 7. 图和完整数据怎么看？', '',
        'figures中SRK/Kashin、1/2bit分别画图。round是准确率随轮次变化；total_uplink_bits是随累计上行变化，横轴为对数；time把在线编码和完整压缩分开。阴影是5组之间的标准差，不是置信区间。Original和全部唯一方法保留。',
        '上行按实际索引位数与元数据计费，没有真实网络传输。固定码本20个float32中心共640bit，给10客户端分发至少6400bit，另有配置成本；这项下行预共享成本没有混进上行。',
        'training_comparison.csv含全部105个seed/方法终点和分项时间；paired_comparisons.json含44项比较、逐seed差值、末五轮/损失差及85/90/92%达标配对状态。独立decode、表示、逆变换、完整压缩和串行算法时间均分开记录。串行算法不含网络、评估和写盘。',
        'verification.md写清楚实际检查了什么、哪些历史细节没有日志可重放。input_manifest.json保存来源和哈希。']
    # Do not hard-code the number of comparison rows in prose.
    lines=[s.replace('含44项比较',f'含{len(comparisons)}项比较') for s in lines]
    (out/'Focused_Fixed_Codebook_Comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
