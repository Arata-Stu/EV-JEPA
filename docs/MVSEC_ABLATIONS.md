# MVSEC ablationと可視化の実行ガイド

## 目的と評価順序

このガイドは、MVSEC Stage 1で次の問いを分離して検証するためのものです。

1. CMax補助損失は、JEPAのrecurrent表現を改善するか。
2. latent trajectoryへの時間正則化は、collapseを悪化させず時間的一貫性を改善するか。
3. 改善はflowだけでなくmetric depthにも移るか。
4. CMax head自身が、GT flowと物理的に整合するか。

実験は次の順で進めます。

```text
outdoor_day2 event-only pretrain
        ↓
outdoor_day2の時系列train/devで条件選択
        ↓
選択した単一runを固定
        ↓
outdoor_day1 final test（必要ならnight1 depth OOD）
```

day2 devは、ラベルを時間前半のprobe学習用と時間後半のdev評価用へ分け、両者の依存event区間の
間にguard gapを置きます。ただしencoderの自己教師あり事前学習自体はday2全eventを見ているため、
これは厳密には**transductiveなlabel-holdout dev**です。真のrecording-level generalizationは、条件を
選び終えた後のday1 final testで判断します。

`outdoor_day1`を多数の条件やweightの選択へ使ってはいけません。評価runnerはこの誤用を避けるため、
final stageで単一の`--selected-run-id`を必須にしています。

## 3つの安全なrunner

| 入口 | 役割 | 既定動作 |
| --- | --- | --- |
| `scripts/experiments/train_mvsec_ablation.sh` | 事前学習条件とseedの生成・実行 | `plan` |
| `scripts/experiments/eval_mvsec_ablation.sh` | day2 devまたはday1 finalのflow/depth評価 | `plan`, `dev` |
| `scripts/experiments/visualize_mvsec_ablation.sh` | report比較、sample描画、CMax診断 | `plan` |

`plan`は出力directoryを作らず、実行予定の条件とcommandだけを表示します。学習を始めるには
`--action run`を明示します。生成configはimmutableで、内容の異なる既存fileを上書きしません。
同じartifact rootへ別suiteを段階的に追加するときは`--skip-complete`を明示します。このoptionは、
completion markerとconfig、checkpoint、manifest、report、head、logのidentity/SHA-256が一致する完了済み
jobだけを再利用します。marker欠落、partial output、引数や内容の相違が一つでもあれば、最初の高コストjobを
始める前に行列全体を停止します。

以下では、前処理済みbundleを`/datasets/evjepa/mvsec`、実験出力を
`/runs/mvsec_ablation`とします。

## 1. 事前学習行列

### 時間正則化の定義

添付案のRate Alignment（RA）とLatent Straightening（LS）は、現モデルにevent-wise logitや
codebookがないため、causal ConvLSTMのrecurrent patch tokenへwindow-levelに適用します。したがって、
Neural Eventsのevent-level手法の再現ではなく、このproject向けの適応です。

patch `n`、window `t`のevent activityを`a[t,n]`、window時間を`dt[t]`とし、supported patchの
rate `r=a/dt`をclip内平均rateで無次元化します。RAのweightは、隣接rateのsymmetric relative差から
次のように作ります。

```text
d_rate = |r[t]-r[t-1]| / (0.5*(|r[t]|+|r[t-1]|) + eps)
w_rate = exp(-gamma*d_rate)
L_RA   = mean(w_rate * mean_D((h[t,n]-h[t-1,n])^2))
```

weightはevent metadataから作ってgradientを切り、両windowに十分なeventがあるpatchだけを使います。
共通のevent-count単位や時間単位を変えてもweightが変わらない設計です。

LSは、3つの連続windowでsupportedなpatchについて次を計算します。

```text
delta_prev = h[t-1,n] - h[t-2,n]
delta_next = h[t,n]   - h[t-1,n]
L_LS       = mean(1 - cosine(delta_prev, delta_next))
```

どちらかの差分normが`eps`以下のtripletは除外します。そのため、LS単体が「全latentを同一にする」
ことで直接rewardを得ることはありません。

既存の`temporal_sigreg_weight`は別の目的です。これはevent-supportでpoolしたlatent一次差分の分布へ
SIGRegをかけるanti-collapse正則化であり、RAのrate条件付きL2でもLSの方向整合でもありません。
実験名とlogも`Temporal SIGReg`、`RA`、`LS`を分離しています。RA/LS suiteでも既定のFrame/support
SIGRegは維持し、collapse-controlそのものを外すのは`frame` suiteだけです。

### 用意したsuite

| suite | 変更する軸 |
| --- | --- |
| `core` | JEPA-only対JEPA+CMax `0.05` |
| `rate_alignment` | RA weight `0, 0.001, 0.01, 0.05`、gamma `1` |
| `rate_gamma` | RA weight `0.01`でgamma `0.5, 1, 2` |
| `straightening` | LS weight `0, 0.001, 0.01, 0.05` |
| `latent` | RA `0/0.01` × LS `0/0.01`の2×2 |
| `latent_cmax` | 選択済みRA+LS `off/on` × CMax `0/0.05`の2×2 |
| `temporal_sigreg` | Temporal SIGReg weight `0, 0.01, 0.02, 0.05` |
| `cmax` | CMax weight `0, 0.01, 0.05, 0.10` |
| `context` | supervised recurrent length `4, 8, 16` |
| `interaction` | Temporal SIGReg `0/0.02` × CMax `0/0.05`の2×2 |
| `reference` | CMax reference `past, future, both` |
| `scales` | CMax temporal scales `[1]`, `[1,2]`, `[1,2,4]` |
| `frame` | Frame/support SIGReg `0`対`0.02` |
| `all` | 上記のCartesian productではなく、重複を除いた25条件の和集合 |

RA/LSの`eps=1e-6`、RA normalization=`per_clip_mean_supported_patch_rate`は全条件で固定し、
生成config metadataにも記録します。weight候補は初期scaleを抑えた探索範囲であり、普遍的な推奨値では
ありません。まずday2 devでraw loss、weighted loss、JEPA loss比、有効pair/triplet数を確認します。

まず最小のJEPA対JEPA+CMaxを確認します。

```bash
bash scripts/experiments/train_mvsec_ablation.sh \
  --action plan \
  --suite core \
  --seeds 0,1,2 \
  --data-root /datasets/evjepa/mvsec \
  --output-root /runs/mvsec_ablation
```

設定だけを確定する場合は`prepare`、実際に順次学習する場合は`run`へ変えます。

```bash
bash scripts/experiments/train_mvsec_ablation.sh \
  --action prepare \
  --suite core \
  --seeds 0,1,2 \
  --data-root /datasets/evjepa/mvsec \
  --output-root /runs/mvsec_ablation

bash scripts/experiments/train_mvsec_ablation.sh \
  --action run \
  --suite core \
  --seeds 0,1,2 \
  --data-root /datasets/evjepa/mvsec \
  --output-root /runs/mvsec_ablation \
  --milestone-epochs 10,25,50,75,100
```

最終epochは必ず最後のmilestoneに含めます。評価は変更され得る`checkpoint-latest.pt`ではなく、
`checkpoint-epoch0100.pt`のような固定checkpointだけを読みます。

大きなjobを開始する前には、独立したsmoke IDで入出力、VRAM、lossの有限性を確認します。

```bash
bash scripts/experiments/train_mvsec_ablation.sh \
  --action run \
  --suite core \
  --seeds 0 \
  --smoke \
  --data-root /datasets/evjepa/mvsec \
  --output-root /runs/mvsec_ablation
```

`all`のような大きい行列を実行するには`--allow-large-matrix`も必要です。一括実行より、coreの
dev結果を確認してから、時間正則化、CMax weight、context length、CMax reference/scaleの順に
段階的に進めることを推奨します。

同じrootでcore完了後にRAを追加する例です。共有baselineは厳密検証後にskipされ、新しい条件だけが
実行されます。

```bash
bash scripts/experiments/train_mvsec_ablation.sh \
  --action run \
  --suite rate_alignment \
  --seeds 0,1,2 \
  --skip-complete \
  --data-root /datasets/evjepa/mvsec \
  --output-root /runs/mvsec_ablation \
  --milestone-epochs 10,25,50,75,100
```

`--skip-complete`は中断runのresume機能ではありません。中断runを継続するときは、そのrun集合だけを
`--resume`で再開します。`--resume`と`--skip-complete`は同時指定できません。

`context`は総supervised-frame exposureを揃えるため、長いsequenceほど1 epochのclip数を減らします。
この比較ではoptimizer update数が変わるため、その制約も結果へ明記します。update数を揃えた補助比較を
行う場合は、frame exposureと計算量が変わる別protocolとして扱います。後段probeでは全条件を
同じ10-window historyで評価し、事前学習context長とdownstream入力履歴を交絡させません。

### seedの扱い

- `--seeds 0,1,2`はencoder事前学習seedです。
- 後段の`--probe-seeds 0,1,2`はrandom probe headの最適化seedです。
- 同じencoder checkpoint内でprobe seedを平均し、その後encoder seed間を比較します。
- 3 encoder seed × 3 probe seedの9本を、独立な9 encoder seedとして扱ってはいけません。

可視化runnerへ複数の`--probe-seeds`を渡すと、この二段階を
`probe mean → encoder mean/std`の順で集約します。条件間でencoder/probe seed集合が一致しない場合は
比較を拒否します。

## 2. day2 devでのflow/depth評価

主比較はcausal入力、native flow horizon、random-init flow/depth probeです。

```bash
bash scripts/experiments/eval_mvsec_ablation.sh \
  --action plan \
  --stage dev \
  --suite core \
  --seeds 0,1,2 \
  --probe-seeds 0,1,2 \
  --pretrain-epoch 100 \
  --tasks primary \
  --protocol-suite primary \
  --save-visualizations 8 \
  --visualization-max-events 200000 \
  --data-root /datasets/evjepa/mvsec \
  --artifact-root /runs/mvsec_ablation
```

計画を確認後、`--action run`へ変えます。job数が安全上限を超える場合は、意図を確認するため
`--allow-large-matrix`が必要です。

core評価後に別suiteを同じrootへ追加する場合も`--skip-complete`を付けます。保存するsnapshot数など
評価条件もcompletion identityに含まれるため、値の異なる既存jobを黙って再利用することはありません。

taskの意味は次のとおりです。

| task | head | 役割 |
| --- | --- | --- |
| `flow-random` | checkpointから独立した固定仕様のrandom head | encoder表現の主比較 |
| `depth` | 同一仕様のrandom log-depth head | 幾何情報の転移比較 |
| `cmax-direct` | 事前学習済みCMax head | zero-shot flowの物理診断 |
| `flow-cmax-init` | CMax headからwarm start | head初期値の参考診断 |

`flow-cmax-init`と`cmax-direct`はCMax checkpointにだけ適用されます。encoder品質の主張には、
JEPA-onlyとJEPA+CMaxの両方へ同じ`flow-random`契約を使います。

protocol suiteは次を生成します。

| suite | 内容 | 扱い |
| --- | --- | --- |
| `primary` | causal + native flow、causal depth | 主結果 |
| `rate` | nativeとdt1 scalar診断 | 補助診断 |
| `alignment` | causalとF³-style centered 50 ms | 因果性診断 |
| `all` | 上記の明示的grid | exploratory |

`dt1`はnative GT flowを22,222 µsへscalar換算する診断で、APS時刻へ再標本化したexact
EV-FlowNet 800-frame protocolではありません。`f3_centered`は未来eventを含むため、主結果には
使いません。

## 3. 選択済みrunのfinal評価

day2 devのmetric、collapse指標、学習安定性、可視化を見て条件を一つ選び、そのrun IDとepochを
実験記録へ固定します。finalではsuite sweepを受け付けません。

```bash
bash scripts/experiments/eval_mvsec_ablation.sh \
  --action plan \
  --stage final \
  --selected-run-id mvsec_jepa_cmax_w0p05__seed0 \
  --probe-seeds 0,1,2 \
  --pretrain-epoch 100 \
  --tasks primary \
  --protocol-suite primary \
  --include-ood \
  --data-root /datasets/evjepa/mvsec \
  --artifact-root /runs/mvsec_ablation
```

このstageだけがday1 final flow/depthと、任意のnight1 depth OODを開きます。計画が正しければ
`--action run`へ変えます。

## 4. 評価sampleの保存と描画

評価時に、再現性情報付きの小型snapshotを保存できます。snapshotは予測を生成した評価jobの中で
保存するため、既存reportへ後から追加できません。上の§2のように、初回の`plan`と`run`へ
`--save-visualizations`を含めてください。すでにsnapshotなしで完了したjobを残す場合は、別の
`--output-root`へ再評価します。

snapshotはpickleを使わない圧縮NPZで、event表現、GT、予測、valid mask、sample/timestamp、
checkpoint・manifest・reference集合のidentityを持ちます。flow snapshotはIWE描画用のraw eventも
上限付きで含みます。既定の選択は、定量集計へ入る有効pixel数を満たしたsampleのうち評価順の
先頭N件です。したがって、定量値の代表標本ではなく定性的な固定標本です。

保存済みsampleとmetricをまとめて描くにはwrapperを使います。

```bash
bash scripts/experiments/visualize_mvsec_ablation.sh \
  --action plan \
  --mode all \
  --stage dev \
  --suite core \
  --seeds 0,1,2 \
  --probe-seeds 0 \
  --pretrain-epoch 100 \
  --task flow-random \
  --alignment causal \
  --dt native \
  --data-root /datasets/evjepa/mvsec \
  --artifact-root /runs/mvsec_ablation
```

`--action run`で、次を生成します。

- flow: event image、GT/pred flow HSV、EPE heatmap、valid mask
- depth: event image、GT/pred metric depth、absolute error、valid mask
- CMax/IWE: warp前、予測flow warp後、GT flow warp後
- 複数run: metric bar、学習curve、seedごとの平均とpopulation standard deviation
- 出力: PNG、HTML、machine-readable JSON

report比較は、protocol、target/reference集合、head仕様などが一致しないrunを既定で拒否します。
`--allow-incompatible`は探索的な図にだけ使い、正式な比較表には使いません。

個別のsnapshotはCLIから直接描けます。

```bash
window-jepa-mvsec-visualize sample \
  --snapshot /path/to/sample.npz \
  --output-dir /path/to/rendered-sample
```

複数reportを明示的に比較する例です。

```bash
window-jepa-mvsec-visualize compare \
  --run jepa__seed0=/path/to/jepa/report.json \
  --run jepa_cmax__seed0=/path/to/cmax/report.json \
  --aggregate-seeds \
  --output-dir /path/to/comparison
```

学習で使ったmultiscale CMax objectiveそのもののzero/shuffled-flow対照、flow field、IWEを確認する
場合は、wrapperの`--mode cmax-raw`または既存の`window-jepa-visualize-flow`を使います。MVSEC
snapshot内のGT-warp IWEは物理的な比較診断であり、学習時CMax lossの厳密な再計算ではありません。
CMax raw reportにも、使用したcheckpointとmanifestのSHA-256・size・file identityを保存し、load中や
描画中の差し替え、および入力と出力先の衝突を拒否します。

RA/LSとJEPAのscale関係は、事前学習`train.jsonl`も直接比較できます。

```bash
window-jepa-mvsec-visualize compare \
  --run jepa__seed0=/runs/mvsec_ablation/pretrain/mvsec_jepa__seed0/train.jsonl \
  --run ra_ls__seed0=/runs/mvsec_ablation/pretrain/mvsec_jepa_ra_w0p01_g1_ls_w0p01__seed0/train.jsonl \
  --curve future_prediction_loss \
  --curve rate_alignment_weighted_loss \
  --curve latent_straightening_weighted_loss \
  --curve prediction_std \
  --output-dir /runs/mvsec_ablation/visualized/pretrain-ra-ls-seed0
```

pretrain JSONL同士には下流評価contractがないため、この図は学習診断です。flow/depthの採用判断は、
contract付きのdownstream report比較で行います。

## 5. 採用・棄却の基準

CMax lossや時間正則化lossが下がっただけでは採用しません。少なくとも次を同時に確認します。

- day2 devのflow AEPE/3PEとdepth AbsRel/SILog
- day2 devのpaired seed差とseed間分散
- JEPA prediction lossとactive/inactive patch別loss
- latent std/rank、Frame/Temporal SIGReg指標、collapse兆候
- CMax valid-window率、flow saturation、zero/shuffled対照
- causal結果が`f3_centered`だけの改善に依存していないこと
- 最終的にday1で同方向の改善が再現すること

最初は`core`だけを3 encoder seedで完走し、主比較が成立してから一軸ずつ広げます。全条件を一度に
回してday2 dev上の偶然の最大値を選ぶと、devへ過適合しやすくなります。

## 6. 再現性artifact

runnerは次を出力pathへ組み込みます。

```text
/runs/mvsec_ablation/
├── configs/<run-id>.yaml
├── pretrain/<run-id>/{checkpoint-epochNNNN.pt,.ablation-complete.json}
├── eval/{dev,final}/<run-id>/epochNNNN/<task>/<protocol>/...
├── logs/{pretrain,dev,final}/...
└── visualized/{compare,samples,cmax-raw}/{dev,final}/<suite-or-selected-run>/...
```

評価reportにはcheckpoint、manifest、GT artifact、target/reference集合、alignment、実際のevent
dependency区間、padding/座標系、head seedを記録します。pairedな単一seed比較ではhead seedを固定し、
複数seed集約では同じencoder/probe seed集合だけを許します。比較時はrun名だけでなく、評価contractと
artifact identityが一致していることを確認してください。

## 参考資料

- [Neural Events: Discrete Asynchronous Autoencoders for Event-Based Vision](https://arxiv.org/pdf/2606.19835)
- [Motion-aware Event Suppression for Event Cameras](https://rpg.ifi.uzh.ch/docs/arxiv26_pellerito.pdf)
- [Fast Feature Fields for Event Camera Vision](https://arxiv.org/html/2509.25146)
- [MVSEC公式download](https://daniilidis-group.github.io/mvsec/download/)
