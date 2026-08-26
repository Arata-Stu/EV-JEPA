# Gen1 Sequence / SIGReg R0 実験計画

更新日: 2026-08-25

## 目的

Event Cameraの50 ms表現について、次の3段階を一度に混ぜず、前段の結果を固定してから
次の軸へ進む。

1. 50 ms内の入力表現を決める。
2. 時系列modelを決める。
3. SIGRegの方式を比較する。

最終的な問いは「JEPA/SIGRegの事前学習特徴がGen1下流タスクへ有効か」であり、
事前学習lossの大小だけで方式を選ばない。事前学習ではGen1のクラス・bbox labelを
使用せず、labelは凍結probeまたはDetection評価でのみ使用する。

## 現在の実装状態

計画と、今すぐ実行できる範囲を区別する。

| 段階 | 比較軸 | 現在の状態 | 実行判定 |
| --- | --- | --- | --- |
| Stage 1 | 50 ms・2 ch / 50 ms・10 ch | `temporal_bins`切替とfeedforward sequence loaderは実装済み | 実行可能 |
| Stage 2 | Feedforward / ConvGRU / ConvLSTM | 3 model、mixed BPTT/TBPTT loader、共有augmentation、stateful Detectionを実装済み | 実行・主評価可能 |
| Stage 3 | SIGReg 3方式 | patch activityの返却だけ実装済み。SIGReg loss、projector、DDP統計は未実装 | まだ実行不可 |

Stage 2では`gen1_detection --stateful`がrecurrent checkpointを受け取り、recording順に
ラベルなし窓を含めてstateを更新する。既存ROI probeは引き続きrecurrent checkpointを拒否する
ため補助分類評価は未実装だが、主指標のstateful Detection APでmodel選択できる。

Stage 3で使用予定の`patch_event_activity: [B,T,P]`はloaderに実装済みだが、
`return_patch_event_activity: true`へ変更しただけではSIGRegは有効にならない。

## 実行server

R0の基準実行環境は`sig-gpu5`に固定する。

| 項目 | 値 |
| --- | --- |
| project root | `/home/iASL/Arata_repo/EV-JEPA` |
| dataset root | `/home/iASL/Arata_repo/dataset` |
| Gen1 root | `/home/iASL/Arata_repo/dataset/gen1_304x240` |
| GPU | NVIDIA V100 × 3 |
| DDP | single-node、3 processes |
| visible devicesの基準 | `CUDA_VISIBLE_DEVICES=0,1,2` |

学習前に`nvidia-smi`で3台が空いていることとGPU indexを確認する。別のindexを使う場合も、
同じ比較とresumeの間は`CUDA_VISIBLE_DEVICES`の順序を固定する。runnerはproject内の相対pathから
起動できるが、datasetは必ず上記Gen1 rootを`--data-root`へ明示する。
project rootはcode checkoutが`/home/iASL/Arata_repo/EV-JEPA`にある前提の例である。実際の
checkout名が異なる場合は`cd`先だけを変更すればよく、runner自身は配置場所からproject rootを
自動解決する。

runner自体はserver構成を固定せず、`--nproc-per-node 1`では単一process、2以上では
single-node DDPを使う。`--nproc-per-node auto`は`CUDA_VISIBLE_DEVICES`で見えているGPUを
すべて使うため、共有serverでは必ず先に利用対象を絞る。GPU数やGPU種類を変えた結果は
R0基準とは別条件であり、同じStage内の比較途中では変更しない。

## 全段階で固定する条件

比較対象以外は次を固定する。

| 項目 | R0の固定値 |
| --- | --- |
| dataset | Gen1、同じtrain/validation manifest |
| event window | 因果窓`(t-50 ms, t]` |
| stride | 50 ms（窓の重複・時間gapなし） |
| supervised sequence | 8 steps |
| recurrent burn-in | 2 steps |
| sampling | per-rankでstream 50% + random 50% |
| patch / image size | 16 / 240×304 |
| backbone | V-JEPA 2.1型 ViT-S、同じdepth・width |
| objective | EMA targetを使うdense Window-JEPA |
| mask | 現行random spatial mask、activity-aware maskなし |
| geometry augmentation | horizontal flipのみ。clip内で共有し、streamではrecording内で共有 |
| optimizer schedule | 100 epochs、10 epochs warmup、同じLR・weight decay |
| supervised signal / epoch | 約50,000 windows |
| precision | FP16（V100用のR0基準。SIGReg統計のみautocast外のFP32） |

入力解像度とcropがともに240×304なので、現在のrandom cropは恒等変換である。
mask、tube mask、augmentation追加、teacherless化はこの3段階では変更しない。

### 再現性の単位

- R0の配線・収束確認は`seed=0`で行う。
- 比較表へ採用する正式結果は`seed in {0,1,2}`の平均と標準偏差を使う。
- dataset manifest、world size、per-rank batch、GPU種類、commit、resolved configを
  runごとに記録する。
- GPU数またはbatch sizeを変更したrunは、同じseedでも別条件として扱う。
- V100 3台では`global_batch = per_rank_batch × 3`である。ConvLSTM・10ch・FP16の実測と
  host-memoryの余裕を基準にper-rank batch 16、global batch 48、worker 4/rankとする。
- gradient accumulationはSIGRegの同時標本数を増やさないため、Stage 3では
  `global_batch = per_rank_batch × world_size`を明記する。

runnerの既定precisionはserver移行に対応するため`auto`とする。`--precision auto`は選択した
全rankの共通能力を調べ、全台native BF16なら`bf16`、全台
Volta以降なら`fp16`、それ以外なら`fp32`へ解決する。異種GPUでは最も低い共通precisionへ
落とす。解決後の具体値はconfig、run ID、metadataへ記録され、`auto`自体は学習configへ
渡さない。

`auto`はserver移行時の安全な初期選択であり、実験条件を自動的に同一にする機能ではない。
新しいGPU構成では最初に独立したsmokeを行い、lossの有限性、checkpoint保存、速度、VRAMを
確認する。その後の正式比較では解決されたworld sizeとprecisionを全条件で固定する。

checkpoint resumeでは、作成時と同じworld size（R0基準では3）、per-rank batch、precision、
resolved configを要求する。runnerはresume時の`auto`を拒否するため、元のrun IDまたは
`launch_metadata.txt`にある具体値を指定する。`--nproc-per-node`だけを1や2へ変えたresume、
またはGPUを1台だけ減らしたresumeは行わず、新しいrunとして最初から実行する。

## 命名規則

run IDは比較軸が名前から分かるようにする。

```text
s{stage}_{input}_{model}_{regularizer}_np{world_size}_bs{per_rank_batch}_{precision}_seed{seed}[_smoke]
```

例えばV100 3台の基準条件は
`s1_input_2ch_ff_nosig_np3_bs16_fp16_seed0`となる。`_smoke`は1 epoch・2 global batchesの
hardware確認専用で、正式runとは別artifactとして保持する。world size、per-rank batch、
precisionをrun IDへ含めることで、互換性のないcheckpointを誤ってresumeしない。

短縮名は次に固定する。

| 軸 | 短縮名 |
| --- | --- |
| 50 ms・2 polarity ch | `input_2ch` |
| 50 ms・5 bin×2 polarity | `input_10ch` |
| Feedforward | `ff` |
| ConvGRU | `cgru` |
| ConvLSTM | `clstm` |
| SIGRegなし | `nosig` |
| Global SIGReg | `sigreg_global` |
| Temporally-Centered SIGReg | `sigreg_tc` |
| Event-Support TC-SIGReg | `sigreg_event_tc` |

出力先は既存の事前学習規約に合わせる。

```text
outputs/pretrain/sequence_sigreg/stage{stage}/{run_id}/
```

同じrun IDの出力が既に存在する場合、実験scriptは上書きせず停止する。resumeは
対応するresolved configとcheckpointを明示した場合だけ許可する。

## Stage 1: 50 ms入力表現

### 研究質問

同じ50 ms分のeventを、極性別に一括積算する方が安定するか、50 ms内の粗い時間配置を
channelとして残す方が有効かを調べる。この段階ではどちらも時系列stateを持たない。

### 比較条件

| run | `temporal_bins` | encoder入力 | 時系列処理 |
| --- | ---: | --- | --- |
| `s1_input_2ch_ff_nosig_np3_bs16_fp16_seedN` | 1 | `[B,T,2,H,W]` | 各50 msを独立に空間処理 |
| `s1_input_10ch_ff_nosig_np3_bs16_fp16_seedN` | 5 | `[B,T,10,H,W]` | 各50 msを独立に空間処理 |

`input_10ch`の各binは10 ms相当だが、5回のrecurrent updateではない。5 bin×2 polarityを
10 channelとして一度にpatch projectionへ渡す。総window長、event集合、sequence sample、
mask、augmentationは2条件で同じにする。

### 選択規則

1. 全seedが100 epochsを完走し、collapse判定に該当しないこと。
2. 主評価は50 msでのGen1下流指標とする。
3. 同じvalidation sample集合で、平均値が高い表現をStage 2へ送る。
4. 差がrun間ばらつき以下なら、channel数が少なく計算量も小さい`input_2ch`を採用する。

Stage 1のfeedforward checkpointは既存の凍結ROI probeとYOLOX Detectionで評価できる。
ROI probeのmacro-F1は短時間の診断、Detection APを最終的な主指標とする。

## Stage 2: 時系列model

### 研究質問

Stage 1で固定した50 ms入力に対し、過去stateを持たないencoder、ConvGRU、ConvLSTMの
どれが有効かを調べる。この段階ではSIGRegを入れない。

### 比較条件

| run | `temporal_model` | objective | state |
| --- | --- | --- | --- |
| `s2_input_{2ch,10ch}_ff_nosig_np3_bs16_fp16_seedN` | `feedforward` | `sequence_dense_window_jepa` | なし |
| `s2_input_{2ch,10ch}_cgru_nosig_np3_bs16_fp16_seedN` | `conv_gru` | `recurrent_dense_window_jepa` | hidden |
| `s2_input_{2ch,10ch}_clstm_nosig_np3_bs16_fp16_seedN` | `conv_lstm` | `recurrent_dense_window_jepa` | hidden + cell |

全条件で同じsequence loaderとloss-bearing 8 stepsを使う。ConvGRU/ConvLSTMは2 stepsの
burn-inでstateを作り、stream laneはbatch境界を越えてdetach済みstateを継承する。
random laneは毎clip resetする。Feedforwardは同じsampleを受け取るが、burn-inとstate
metadataをmodel計算に使わない。

これは「同じ現在frameだけを見た公平な計算量比較」ではなく、時間履歴を利用できるmodelの
実運用上の比較である。parameter数、peak VRAM、supervised windows/sも併記する。
必要なら全モデル`burn_in_steps=0`の計算量寄り感度分析を後から行うが、R0の主表には混ぜない。

### Stage 2の評価protocol

事前学習lossはarchitecture間で表現尺度が異なり得るため、loss最小だけで勝者を選ばない。
正式な選択には、実装済みstateful Detectionが次を満たすことを各runで確認する。

- recording内のtimestamp順で全50 ms窓を入力する。
- labelのない窓でもstateを更新する。
- sequence境界だけでstateをresetする。
- validation中にstreamをshuffleしない。
- Feedforwardにも同一timestamp/sample集合を使う。
- 凍結headとfine-tuneの結果を分ける。

主指標はstateful Gen1 Detection APとする。凍結分類macro-F1はrecurrent版が未実装なので
現時点では補助表から外す。stateful Detectionはbatch 1・凍結backboneで実行し、validation AP
最大時の`checkpoint-best.pt`と最新再開用`checkpoint-latest.pt`を分けて保存する。

### 選択規則

1. 全seedで100 epochsを完走し、collapseしないこと。
2. stateful Detection APのseed平均を主順位とする。
3. 同等ならmacro-F1、安定性、計算量の順で判断する。
4. ConvGRUとConvLSTMが同等なら、stateとparameterが小さいConvGRUを優先する。

## Stage 3: SIGReg

### 研究質問

Stage 2で選んだmodelに対し、通常の潜在分布、時間変化成分、eventが存在する時空間supportの
どこへSIGRegを適用するのがEvent Cameraに適するかを調べる。

SIGRegなしを対照に含めるため、実験条件は3種類ではなく4条件になる。

| run | regularizer | 対象 |
| --- | --- | --- |
| `s3_input_X_MODEL_nosig_np3_bs16_fp16_seedN` | なし | EMA-JEPA baseline |
| `s3_input_X_MODEL_sigreg_global_np3_bs16_fp16_seedN` | Global SIGReg | 各時刻のframe latent |
| `s3_input_X_MODEL_sigreg_tc_np3_bs16_fp16_seedN` | TC-SIGReg | clip内時間中心を引いたframe residual |
| `s3_input_X_MODEL_sigreg_event_tc_np3_bs16_fp16_seedN` | Event-Support TC-SIGReg | active patchの時間residual |

ここで`X`はStage 1で選んだ`2ch`または`10ch`、`MODEL`はStage 2で選んだ
`ff`、`cgru`、`clstm`のいずれかである。Stage 3でbatchを6へ増やした場合はrun IDも
`bs6`となる。

### Stage 3で固定する仕様

- EMA targetとJEPA prediction lossは残す。`shared_online` teacherless化は別の比較軸とする。
- 3方式で同じprojector構造、projector次元、projection数、knots、loss weightを使う。
- projector入力にはfull/unmaskedなonline encoder latentを使い、時刻ごとに変わるJEPA maskを
  時間変化として学習させない。
- recurrent modelのfull/unmasked補助branchは予測branchとstateを分離し、現在のunmasked
  patchがJEPA予測stateへ漏れないようにする。
- SIGRegの計算はautocast外のFP32とする。
- random projectionは全rankで一致させる。
- 各時刻でglobal batch方向の統計を取り、`B×T`を独立標本としてflattenしない。
- DDP集約はforward値だけでなくgradientもsingle-process参照と一致させる。
- TCの時間窓はloss-bearing clip内に閉じ、burn-in、state reset、TBPTT detach境界を跨がない。
- `temporal_window=1`は時間残差が常に0になるため設定errorとする。
- Stage 3の全条件で同じglobal batchを使う。Stage 2 baselineとbatchが異なる場合、
  `nosig`もStage 3のbatchで再実行する。

Global SIGRegとTC-SIGRegはframe latentを対象とする。Event-Support TC-SIGRegのR0仕様は
patch token `h[b,t,p,d]`とraw event count `a[b,t,p]`から、まずactivity-awareなframe
latentを作り、その後はTC-SIGRegと同じ時刻別batch統計を使う。

1. `support[b,t,p] = (a[b,t,p] > 0)`を作る。
2. `v[b,t,d] = sum_p support * h / max(sum_p support, 1)`としてactive patchを空間poolingする。
3. 各sequenceのloss-bearing時間窓内で`v`の時間中心を引き、frame residualを作る。
4. 各時刻でglobal batch方向へ通常のSIGRegを適用する。
5. active patchがないsequence/timeはskipし、zero-support率を記録する。

これにより3方式でSIGReg estimatorと標本軸を揃え、patch相関を独立標本として数えない。
`log1p(event_count)`によるpooling、activity-weighted temporal mean、active thresholdの調整、
active patchへ直接適用するsequence-balanced ECFは内部ablationであり、最初の3方式比較には
入れない。event-rich sequenceが統計を独占する`B×T×P`の単純flattenは、いずれの方式でも
行わない。

### Stage 3の実装gate

次が揃うまで実験scriptはStage 3を実行してはならない。

- SIGReg Epps–Pulley lossと専用projector
- Global / TC / Event-Support TCのconfig schema
- global-batch DDP統計とgradient一致test
- Gaussian入力、定数入力、非Gaussian入力に対するloss性質test
- TCのsequence定数offset不変性test
- JEPA maskだけを変えてもSIGReg latentが変わらないことのtest
- recurrent補助branchのunmasked入力が予測branch stateへ漏れないことのtest
- 時刻別batch統計であり`B×T` flattenでないことのtest
- Event-Supportのzero-event、sequence均等重み、NaN回避test
- optimizer・checkpoint・resumeへのprojector接続
- 次節のSIGReg診断値のlogging

SIGRegのglobal batchが小さ過ぎると統計が不安定になる。gradient accumulationではなく
同じforward内のglobal batchを使い、最初のscaling確認では少なくとも16、可能なら32以上を
確保する。正式値は全Stage 3条件で固定して結果表へ記載する。

基準のper-rank batch 16・V100 3台ではglobal batchは48である。SIGRegの統計には従来の
global batch 12より適しているが、正式実験ではglobal sample数と有効標本数を必ず記録する。
`nosig`を含むStage 3の全4条件は同じbatchで最初から実行し、Stage 1/2の途中でも一条件だけ
batchを変えて比較しない。

## 記録する指標

### 全段階

- JEPA total / masked / dense / deep-supervision loss
- prediction std、target std
- gradient norm
- mask active-patch率、event-mass coverage、empty-target率
- recurrent state RMS
- epoch時間、supervised windows/s、peak GPU memory
- trainable parameter数
- Gen1 frozen ROI macro-F1（診断）
- Gen1 Detection AP / AP50 / AP75 / AP-S / AP-M / AP-L（主評価）

### SIGReg実装後に追加

- prediction lossとSIGReg lossを分離した値
- `lambda * sigreg_loss / prediction_loss`
- projector出力の平均、標準偏差、effective rank
- temporal residualの標準偏差、effective rank
- SIGReg effective sample数
- Event-Supportのactive-patch率、zero-support率
- ECFのreal/imag誤差

学習lossは収束・崩壊診断に使い、最終順位は同一protocolの下流validation指標で決める。
test splitを方式選択に使わない。

## 実行順

各Stageで次の順序を守る。

1. configを生成し、比較軸以外のresolved差分がないことを確認する。
2. 時系列inspection HTMLを保存し、窓の連続性、`loss_mask`、state reset、augmentation共有を
   目視確認する。
3. `seed=0`で独立したsmoke runを行い、3-rankのforward/backward/checkpoint保存を確認する。
4. `seed=0`の100-epoch R0を全条件で完走させ、崩壊・資源量を確認する。
5. 正式比較では`seed=1,2`も同じ100 epochsで実行する。
6. 同一sample集合で下流評価し、平均・標準偏差をまとめる。
7. 選択規則を満たした条件名とcheckpoint hashを`selected_*.txt`へ固定して次Stageへ進む。

`--smoke`は通常runと別IDであり、resume対象にしない。通常runが中断した場合だけ、同じ
world size・per-rank batch・precision・resolved configでresume契約を確認する。

100 epochs時点でもlossが改善している可能性があるため、相対lossの停滞を理由に早期終了しない。
全条件を同じepoch数で比較し、100 epochs後の延長学習は選択後の別runとする。

## 停止条件

次の場合は直ちにrunを停止し、出力を消さず`failed`として記録する。

- loss、gradient norm、model stateにNaN/Infが出た。
- DataLoaderのtimestamp順、stream ID、augmentation ID、state reset契約が破れた。
- checkpointのresolved configまたはhashが実行条件と一致しない。
- OOM回避のため、そのrunだけbatch、画像サイズ、sequence長を変更する必要が生じた。
- Stage 3でglobal ECFの有効標本が不足、またはEvent-Supportが全てzero-supportになった。

`prediction_std`または`target_std`が3 epoch連続してほぼ0になる場合はcollapse候補として
停止できる。ただし固定閾値はprojector実装・初期scaleに依存するため、Stageごとのsmokeで
一度だけ決め、全条件へ同じ値を適用し、結果表へ明記する。悪い下流精度やlossの改善速度だけを
理由に途中で条件を打ち切らない。

OOMで条件を変える場合は、その条件だけ再開せず、比較する全runを新しい共通条件で最初から
やり直す。例えばper-rank batch 16から12へ下げるなら、そのStageの全比較条件をbatch 12・
global batch 36で新しいrunとして実行する。既存runは別IDで保持し、OOMしたcheckpointから
batch sizeを変えてresumeしない。

## 実験script

実行入口は次に統一する。

```bash
bash scripts/experiments/run_sequence_sigreg_plan.sh --help
```

scriptはconfig準備、run一覧表示、inspection、事前学習をStage別に行い、既存runへの
上書きを拒否する。Stage 3は実装gateが満たされるまで、未対応configを生成したふりをせず
明示的に停止しなければならない。

対応する実行指定は次のとおりである。

| 指定 | 動作 |
| --- | --- |
| `--nproc-per-node 1` | launcherを介さない単一GPU実行 |
| `--nproc-per-node N`（`N >= 2`） | `N` processのsingle-node DDP |
| `--nproc-per-node auto` | 可視GPU数を使用。事前に`CUDA_VISIBLE_DEVICES`で対象を限定 |
| `--precision fp32` | autocastなしの基準経路 |
| `--precision fp16` | FP16 autocastとgradient scaling。V100/Volta以降向け |
| `--precision bf16` | BF16 autocast。選択した全GPUのnative BF16を必須化 |
| `--precision auto` | 全rank共通で`bf16`、`fp16`、`fp32`の順に安全な候補を選択 |

FP16では動的gradient scalingを用いる。TBPTTの途中ではscaleを変更せず、全chunkのbackward後に
一度だけunscaleとgradient clipを行う。非有限gradientが検出された更新ではoptimizerとEMAを全rankで
同時にskipし、loss scaleを下げる。`loss_scale`と`optimizer_step_skipped`を記録し、GradScaler状態を
checkpointへ含める。`attempt_step`はbatch試行数、`global_step`は成功したoptimizer更新数とする。
FP32はautocastを使わない基準経路として常に残す。

再開用`checkpoint-latest.pt`は`checkpoint_every_epochs: 1`により毎epoch更新する。最良の
pretrain重みを後から下流評価できるよう、runnerは既定で5 epochごとと最終epochを
`checkpoint-epochNNNN.pt`として保持する。保持間隔は`--keep-every-epochs`で明示し、異なる条件間で
揃える。これらはoptimizer等を含む完全checkpointなので必要容量も実験開始前に確認する。

例えばV100 1台のportable smokeは次のように開始する。実行時には`auto`が
`--nproc-per-node 1 --precision fp16`相当へ解決され、run IDも具体値を使う。

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 1 --action all --seed 0 --smoke \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --nproc-per-node auto --precision auto --batch-size 16 --workers 4
```

`sig-gpu5`では、最初に実行予定の全体像だけを表示する。以下の例はV100 3台で`auto`を
FP16へ解決し、per-rank batch 16（global batch 48）、worker 4/rankを使う。

```bash
cd /home/iASL/Arata_repo/EV-JEPA
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage ready --action plan \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto
```

次に、正式runを作る前の必須手順として、同じ3 GPUでStage 1のhardware smokeを実行する。
`--smoke`は各条件を1 epoch・2 global batchesだけ実行し、通常runとは別の`_smoke`付きrun IDを
使う。smokeはresumeせず、失敗時は原因を修正して新しい出力先でやり直す。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 1 --action all --seed 0 --smoke \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto
```

smokeでは、3 rankが起動すること、各GPUへmodelが配置されること、forward/backward、
checkpoint保存、時系列inspectionが成功することを確認する。smokeのloss値は方式選択には
使用しない。Stage 2へ進んだ際も、選択したinputで3 modelの`--smoke`を先に完走させてから
100-epoch runを開始する。

Stage 1は、config生成、時系列inspection、2条件の事前学習を別々のactionで実行する。
`--action all`を指定すれば、この3操作を順番にまとめて実行できる。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 1 --action prepare --seed 0 \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto

CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 1 --action inspect --seed 0 \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto

CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 1 --action run --seed 0 \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto
```

Stage 1で10 chを選んだ例では、Stage 2を次のように開始する。2 chを選んだ場合は
`--selected-input 2ch`へ変更する。まず3 modelのsmokeを完走させる。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 2 --action all --selected-input 10ch --seed 0 --smoke \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto
```

その後、同じ条件で正式runを開始する。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 2 --action all --selected-input 10ch --seed 0 \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision auto --batch-size 16 --workers 4 --nproc-per-node auto
```

中断後に同じworld size・configで再開する場合だけ、明示的に`--resume`を付ける。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
  bash scripts/experiments/run_sequence_sigreg_plan.sh \
  --stage 2 --action run --selected-input 10ch --seed 0 \
  --data-root /home/iASL/Arata_repo/dataset/gen1_304x240 \
  --precision fp16 --batch-size 16 --workers 4 --nproc-per-node 3 --resume
```

resume時は、checkpoint作成時の`CUDA_VISIBLE_DEVICES`、`--nproc-per-node 3`、
`--precision fp16`、`--batch-size 16`、`--workers 4`を変更しない。world sizeやbatchを変えたい場合は
`--resume`を付けず、新しいrun IDと共通条件でそのStage全体をやり直す。

正式な3 seed比較では、各Stageの選択条件を固定した後に`--seed 0`、`1`、`2`を個別に
実行する。`stage=ready`は計画表示専用であり、Stage 1の結果を見ずにStage 2まで一括実行する
ことは許可しない。

macOSは実行環境にせず、config生成・shell構文確認・差分確認までとする。`auto`はCUDA対応
PyTorchを使ってGPUを調べるため、macOSでplan/prepareする場合はGPU数とprecisionを具体的に
指定する。GPU学習、PyTorchへ依存するcheckpoint確認、下流評価はserver PCのproject rootで
行う。

## 今回は固定・延期する軸

以下も研究上は重要だが、Stage 1〜3へ混ぜると要因を分離できないため延期する。

- EMA target / shared-online teacherless
- 同一時刻masked latent / future latent prediction
- JEPA / MAE
- stepごとのmask / tube mask
- 10 msを5 recurrent stepsとして処理する方式
- BPTT/TBPTT比、burn-in長、history長
- random crop、rotation、scale、時間augmentationの追加
- SIGReg weight、TC window長、activity weightのsweep
- Stage 3後のFeedforward / ConvGRU / ConvLSTM × SIGReg交差比較

まずStage 1〜3で有望な構成を得てから、上記を独立したablationとして追加する。
