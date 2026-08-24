# Event Window-JEPA

異なるイベント蓄積時間の間を潜在空間で予測し、未学習の蓄積時間にも頑健なイベント表現を学習するための独立実装です。

この版の成果物は、研究全体を完了したend-to-end実装ではなく、**Window-JEPA事前学習コアとmatched-window評価protocolのscaffold**です。Gen1/DSEC/MVSECの公式下流pipelineが未接続のため、現時点のコードだけで論文の成立条件を実証したとは扱いません。

中心となる処理は次のとおりです。

```text
同じ終了時刻 t
  ├─ X(t, Δc) ─ Online ViT ─ context tokens ─┐
  │                     s(Δc)                 ├─ Window Predictor ─ Ẑ(t, Δt)
  └─ X(t, Δt) ─ EMA ViT ─ target tokens ─────┘      s(Δc), s(Δt), s(Δt/Δc)
```

画素やイベントそのものは復元せず、EMA encoderが出力するtarget patch特徴だけを予測します。推論時にはpredictorへ40 msなどの基準時間を問い合わせ、全patchのcanonical latentを取得できます。

## 現在の実装範囲

実装済みです。

- sequence単位のJSONL manifestとNPZ/HDF5 EventStore
- 厳密な因果窓 `(t - Δ, t]` の二分探索
- 時間軸上一様のanchor sampling
- 同一終了時刻のcontext/target window pair
- context/targetで共有されるcropとhorizontal flip
- 2極性 × 5 bin、時間線形補間、窓境界基準、要素ごとの`log1p` voxel grid
- 2チャネルON/OFF event image ablation
- ViT-S/16相当のonline/EMA encoder
- V-JEPA 2.1型のflat global ViT（2-D RoPE、SDPA、中間層監督）
- 50 ms連続clip用のsequence samplerとclip共有geometry
- patch-grid ConvLSTM / ConvGRU、burn-in、BPTT＋TBPTT学習
- `log(Δ)` Fourier embedding
- `Δc`、`Δt`、`log(Δt/Δc)`で条件付けたcross-attention predictor
- disjoint multiblock spatial mask
- Smooth L1 latent loss、任意のVICReg variance補助loss
- encoder-only / canonical latentの両出力
- 単一GPUおよび`torchrun`によるDDP事前学習
- atomic checkpoint、厳密resume、collapse診断値のJSONL記録
- higher-is-better / lower-is-better双方に対応したwindow robustness集計
- fixed-window JEPA、直接feature consistency、Window-JEPAの設定例
- DSEC・M3ED・RVT Gen1/Gen4・Prophesee DATからZstd HDF5へのstreaming前処理

下流タスク固有のデータ変換・head・公式evaluatorは、このリポジトリへ外部コードを直接コピーせず、`event_window_jepa.downstream.features`の共通feature境界へ接続する設計です。Gen1検出、DSEC segmentation、MVSEC depth/flowの統合は各データセットを用意した段階で追加します。

## セットアップ

想定環境はPython 3.11です。

```bash
python3.11 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,hdf5]'
```

GPU用PyTorchを使う環境では、利用するCUDAに合うPyTorchを先に導入してからeditable installしてください。macOS上で学習を実行することは前提にしていません。

## データ形式

### Manifest

1行1sequenceのJSONLです。splitは必ずsequence単位で分けます。

```json
{"sequence_id":"gen1__train_0001__left","source_recording_id":"gen1__train_0001","path":"../events/gen1_train_0001.h5","group":"/","events_group":"events","height":240,"width":304,"t_start_us":0,"t_end_us":59999999,"split":"train","dataset":"gen1","camera":"left","source_time_origin_us":123456789,"coordinate_frame":"distorted","timestamp_reference":"DAT event/annotation recording clock","timestamp_synchronized":true}
```

相対`path`はmanifestのあるディレクトリから解決されます。`t_start_us`は最初のイベント時刻でなく、そのsequenceで利用可能な時間範囲の下端として扱います。
元時計、camera、歪み座標系、source/stored解像度、整数downsample倍率と方式も各行に
保持するため、下流ラベルを同じ時計・座標へ明示的に変換できます。M3EDとGen4/1Mpxの
factor 2は、既定でDAGR式`area_accumulate`を使い、event数と保持率も記録します。

### HDF5（標準）

大規模な事前学習設定では`data.store: hdf5`を使います。manifestの`group`
（既定`/`）と`events_group`（既定`events`）が指すgroupの下へ、同じ長さの
1次元dataset `x/y/t_us/polarity`（`t/p`も可）を配置します。初期化時に整数
dtypeを要求します。converterは配列長、時刻の全体昇順、座標、極性、manifest境界を
変換時に全件検証します。学習時は完了schemaを軽量検査し、1 ms coarse index内だけを
厳密検索します。ファイルhandleとindexはDataLoader workerごとのLRUです。

DSEC・M3ED・Gen1・Prophesee 1Mpxの具体的な変換方法、解像度、時計、座標系、
storage上の注意は[docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md)を参照してください。
既に1Mpx raw DATを持っている場合は、factor 2を固定し、event・bbox・相対manifestを
移動可能なbundleへまとめる
[1Mpx portable wrapper](scripts/preprocess/preprocess_prophesee_1mpx.sh)を使えます。
データセットの段階的な選定、公式配布元、再開可能なdataset別download script、
手動フォームが必要なケースは[docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md)にまとめています。

### NPZ（小規模確認用）

各NPZには同じ長さの1次元整数配列を保存します。

| key | 内容 | 推奨dtype |
| --- | --- | --- |
| `x` | pixel x座標 | `int16`/`int32` |
| `y` | pixel y座標 | `int16`/`int32` |
| `t_us` | 昇順の整数microsecond時刻 | `int64` |
| `polarity` | `{-1,+1}`または`{0,1}` | `int8` |

NPZは参照のたびにsequence全体を展開する形式なので、少数sequenceのsmoke testや
十分に小さくshard化したデータだけを対象にします。多数sequence・10万sample/epoch
の標準実験では使わず、上記HDF5へ変換してください。

## 事前学習

標準のWindow-JEPA設定は[configs/pretrain/window_jepa_vits.yaml](configs/pretrain/window_jepa_vits.yaml)です。`data.manifest`を実データへ変更して実行します。

```bash
window-jepa-pretrain --config configs/pretrain/window_jepa_vits.yaml
```

Gen1をV-JEPA 2.1型backboneで最初から学習し直す設定は
[configs/pretrain/window_jepa21_vits_gen1.yaml](configs/pretrain/window_jepa21_vits_gen1.yaml)です。
これはConv2d patch projectionの直後をflatなglobal ViTへ入力し、learned absolute
position embeddingの代わりに2-D RoPE、attentionにはSDPAを使います。
`deep_supervision_layers: [2, 5, 8, 11]`は同じ15×19 token gridの中間出力であり、
feature pyramidや階層型ViTではありません。保持context patchから全patchを予測し、
visible/maskedの両方を含むdense lossを4深度で平均します。

```bash
cd ~/Arata_ws/EV-JEPA
PYTHONUNBUFFERED=1 window-jepa-pretrain \
  --config configs/pretrain/window_jepa21_vits_gen1.yaml
```

### Recurrent R0（ConvLSTM / ConvGRU）

R0は50 msの連続した因果窓を同一sequenceから読み、patch projection後の空間gridを
ConvLSTMまたはConvGRUで更新します。online encoderだけが過去stateを継続し、EMA
target encoderは各stepでzero-stateへresetして、現在のunmasked 50 ms窓をtarget
latentに変換します。current context maskはrecurrent convolutionより前に適用される
ため、現在のmasked patchがstate経由でonline側へ漏れることはありません。

標準設定は
[recurrent_r0_convlstm_vits_gen1.yaml](configs/pretrain/recurrent_r0_convlstm_vits_gen1.yaml)
です。

```bash
window-jepa-pretrain \
  --config configs/pretrain/recurrent_r0_convlstm_vits_gen1.yaml
```

この設定ではper-rank batch 4を、RVTと同じ比率でstream 2＋random 2へ分けます。
各itemは次の10窓です。

```text
2 burn-in steps（lossなし、no-grad）
        ↓ stateをdetach
8 supervised steps（batch内BPTT）
        ↓
optimizer / EMAを1回更新
        ├─ stream rows: stateをdetachして同じlaneの次batchへ継承（TBPTT）
        └─ random rows: stateを破棄し、次batchもzero-stateから開始（BPTT）
```

DataLoader出力は`x: [B,T,C,H,W]`で、cropとhorizontal flipはclip全体で共有されます。
各窓は`(t-50 ms,t]`、strideも50 msなので、隣接窓の境界eventは重複も欠落も
しません。stream rowsでは同じrecordingの連続chunkを同じbatch laneへ供給し、cropと
flipもrecording全体で固定します。sequence境界だけでstateをresetします。random rowsは
clipごとに別augmentationを抽選し、毎batch stateをresetします。checkpointには
ConvLSTM/ConvGRUの学習済みweightを保存しますが、一時的なhidden/cell stateは
保存しません。checkpointがepoch境界だけなのはこのためです。

現時点のdata augmentationはrandom cropとhorizontal flipだけです。Gen1 R0では入力
解像度とcropがともに240×304なのでcrop座標は常に`x0=0, y0=0`となり、実際に確率的な
変換は`p=0.5`のhorizontal flipだけです。回転、拡大縮小、時間順序のshuffleは行いません。
VoxelGrid化と`log1p`は決定論的なrepresentation処理で、augmentationには数えません。

JEPAのcontext/target maskは空間augmentationとは別で、現在はstepごとに独立抽選です。
これはR0の収束確認には使えますが、過去にvisibleだったpatchをstateが保持する効果を
切り分けるため、最終比較ではtube maskも対照実験に含めます。

recurrent設定での`data.samples_per_epoch`はwindow数ではなく**clip数**です。標準設定の
6250 clips × 8 supervised stepsは、1 epochあたり約50,000 supervised windowsの
公称値です。実際は全rankで完全なglobal batchだけを使うため端数を切り捨てます
（1 GPU・batch 4なら6248 clips、49,984 supervised windows）。
時系列tensorは大きいため、feedforward設定のbatch size 64をそのまま流用せず、まず
標準設定のbatch size 4からGPU memoryに合わせて調整してください。

下流のstreaming推論ではstateをsequenceごとに所有し、境界で`None`へresetします。

```python
from event_window_jepa.downstream.features import extract_recurrent_patch_features

state = None
for x_50ms in causal_windows:
    tokens, state = extract_recurrent_patch_features(
        model,
        x_50ms,
        duration_ms,
        state,
    )
```

R0では`canonical_query_weight=0`なので、recurrent checkpointの主評価には
`canonical_latent()`ではなく、このstateful encoder出力を使用します。
既存のGen1 ROI probeとYOLOX検出はframe独立のDataLoaderであり、時系列stateを
正しく更新できないため、recurrent checkpointを明示的に拒否します。R0を評価する
ときは、sequence順に全50 ms窓を入力し、ラベルのない中間窓でもstateを更新して、
sequence境界だけで`None`へresetする評価経路を用意してください。

checkpoint、`train.jsonl`、TensorBoardは外部SSDではなく、project内の
`outputs/pretrain/vjepa21_vits_gen1_seed0/`へ保存されます。

学習前に、同じ設定とサンプリング処理で context/target、時間bin、patch maskを
目視確認できます。レポートは追加の描画ライブラリを使わない自己完結HTMLです。

```bash
window-jepa-inspect \
  --config configs/pretrain/window_jepa_vits.yaml \
  --expected-dataset gen1 \
  --samples 8 \
  --output outputs/gen1-inspection/samples.html
```

HTMLと同じ場所に、各整合性検査の結果を含む`samples.json`も保存されます。

Recurrent R0では、連続50 ms窓をstepごとに検査する専用レポートを使用します。

```bash
window-jepa-inspect-recurrent \
  --config configs/pretrain/recurrent_r0_convlstm_vits_gen1.yaml \
  --expected-dataset gen1 \
  --sample-index 0 \
  --output outputs/gen1-inspection/recurrent-clip.html
```

HTML・JSONに加え、各stepのevent画像、representation、時間bin、mask overlayが
`recurrent-clip_assets/`へPNGとして保存されます。時系列・sequence境界・共有crop/flip・
burn-in/TBPTT maskのいずれかが不整合なら、CLIはレポート保存後に終了コード1を返します。
`recurrent.sampling: mixed`の設定では、実際のsamplerから連続する2 batchを取り出して
rank 0相当の経路を検査します。R0のper-rank batch 4では`--sample-index`はbatch内rowを表し、0・1が
stream、2・3がrandomです。HTMLのmixed sampler表とJSONの`mixed_batches`には、
batch間timestamp、state reset、augmentation ID、crop・flipをrowごとに記録します。
stream rowがresetなしで継続するのに時刻・ID・transformが変わった場合、連続した
streamを誤ってresetした場合、random rowが毎batch resetされない場合も終了コード1に
なります。短いsequenceの終了やrecording切替による正当なresetは境界として表示され、
不合格にはしません。
DDP学習時は、これと同じcontinuity/reset契約を各rankの実DataLoader batchに対して
学習ループ内でも毎回検証します。

学習中はrank 0だけにepoch単位の進捗バーを表示し、loss、prediction/target std、
learning rateだけを簡潔に更新します。JSONLの完全な記録は従来どおり
`OUTPUT_DIR/train.jsonl`へ保存し、TensorBoardにはobjective、collapse診断、学習率を
記録します。Dense objectiveでは`loss/dense`、`loss/visible`、
`loss/deep_supervision`も追加されます。

- `loss/total`, `loss/masked`, `loss/canonical`
- `representation/prediction_std`, `representation/target_std`
- `optimization/learning_rate`

TensorBoardは次のように起動します。

```bash
tensorboard --logdir outputs/window_jepa_vits/tensorboard --port 6006
```

複数GPUでは次の形です。

```bash
torchrun --standalone --nproc-per-node=4 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/window_jepa_vits.yaml
```

比較用設定は以下です。

- [fixed_jepa.yaml](configs/pretrain/fixed_jepa.yaml): 40 ms固定・時間条件なし
- [direct_consistency.yaml](configs/pretrain/direct_consistency.yaml): 異なる窓のglobal featureを直接一致し、variance/covariance項でcollapseを防止
- [unconditioned_window_jepa.yaml](configs/pretrain/unconditioned_window_jepa.yaml): B5に対応する時間条件なしcross-window JEPA
- [window_jepa_vits.yaml](configs/pretrain/window_jepa_vits.yaml): 異なる窓の条件付きpatch latent prediction
- [window_jepa21_vits_gen1.yaml](configs/pretrain/window_jepa21_vits_gen1.yaml): V-JEPA 2.1型global ViTとdense/deep-supervision lossによるGen1事前学習

`variance_weight`は既定で0です。`train.jsonl`の`prediction_std`と`target_std`でcollapseを確認した後にだけ有効化します。

## 下流feature

```python
from event_window_jepa.downstream.features import (
    extract_patch_features,
    tokens_to_feature_map,
)
from event_window_jepa.train.checkpoint import load_pretrained_model

model, experiment_config = load_pretrained_model(
    "outputs/window_jepa_vits/checkpoint-latest.pt",
    device="cuda",
)

tokens = extract_patch_features(
    model,
    x=voxel_batch,
    duration_ms=window_ms,
    mode="canonical",
    canonical_ms=40.0,
)
feature_map = tokens_to_feature_map(tokens, model.online_encoder.grid_size)
```

`mode="encoder_only"`との比較で、predictorを蓄積時間変換器として使う効果を切り分けられます。canonical経路はtarget event、target event数、target voxelを一切入力に取りません。
標準設定ではequal-window pairも一部学習し、40→40を含むpredictorのidentity変換を較正します。また、masked patch lossに加えて同じlatent lossをfull-context/full-queryにも0.25の重みで適用し、canonical推論の単一passと学習時の条件を一致させます。この重みは`canonical_query_weight`でablationできます。

### Gen1 frozen ROI probe

Gen1のbbox位置でpatch tokenを平均し、凍結したbackbone上で線形分類器だけを学習する診断用probeを用意しています。これは位置推定を正解bboxに依存するため、公式DetectionのmAPではありません。事前学習済み特徴が物体クラスを分離できるか、また窓長を変えたときに性能が保たれるかを短時間で確認する用途です。

まず40 msだけで少数フレームを試します。

```bash
python -m event_window_jepa.downstream.gen1_roi_probe \
  --checkpoint /path/to/checkpoint-latest.pt \
  --train-manifest /path/to/gen1_304x240/manifests/train.jsonl \
  --val-manifest /path/to/gen1_304x240/manifests/val.jsonl \
  --output-dir /path/to/runs/gen1_roi_probe \
  --mode encoder_only \
  --eval-window-ms 40 \
  --max-train-frames 20000 \
  --max-val-frames 5000
```

問題なく完走したら、上限を外してwindow sweepを実行します。抽出featureはoutput directory以下へfloat16でcacheされるため、同じチェックポイント・同じサンプルでの再実行では再計算しません。

```bash
python -m event_window_jepa.downstream.gen1_roi_probe \
  --checkpoint /path/to/checkpoint-latest.pt \
  --train-manifest /path/to/gen1_304x240/manifests/train.jsonl \
  --val-manifest /path/to/gen1_304x240/manifests/val.jsonl \
  --output-dir /path/to/runs/gen1_roi_probe_encoder \
  --mode encoder_only \
  --eval-window-ms 5 10 15 20 30 40 60 80 120
```

`--mode canonical`でも別output directoryへ実行すると、predictorによる40 ms canonical化の効果を比較できます。主要な結果は`summary_<mode>.json`、窓ごとのmacro-F1は`window_metrics_<mode>.jsonl`へ保存されます。

事前学習済みbackboneの寄与は、同じarchitectureをランダム初期化した対照と比較します。

```bash
python -m event_window_jepa.downstream.gen1_roi_probe \
  --checkpoint /path/to/checkpoint-latest.pt \
  --train-manifest /path/to/gen1_304x240/manifests/train.jsonl \
  --val-manifest /path/to/gen1_304x240/manifests/val.jsonl \
  --output-dir /path/to/runs/gen1_roi_probe_random \
  --mode encoder_only \
  --backbone-init random \
  --eval-window-ms 40 \
  --max-train-frames 20000 \
  --max-val-frames 5000 \
  --no-cache
```

### Gen1 YOLOX Detection

実Detectionでは、全304x240画面を256x320へzero-padし、ViTの位置埋め込みを16x20 patch gridへ補間します。そのtokenからstride 8/16/32のfeature pyramidを作り、プロジェクトに同梱したRVT版YOLOX headを学習します。評価も同梱したProphesee COCO evaluatorを使い、小boxと各recording先頭0.5秒を公式protocolどおり除外します。必要部分はライセンス表示付きで固定しているため、外部RVT repositoryのcloneは不要です。

これは正解bboxを推論入力に使わず、予測bboxからmAPを計算します。一方、RVTのrecurrent backboneや21-step sequenceは使わないため、RVTそのものの再現実験ではありません。まず凍結backboneで事前学習特徴を評価し、必要な場合だけ`--unfreeze-backbone`でfine-tuneします。

```bash
python -m pip install -e '.[hdf5,detection]'

python -m event_window_jepa.downstream.gen1_detection \
  --checkpoint /path/to/checkpoint-latest.pt \
  --train-manifest /path/to/gen1_304x240/manifests/train.jsonl \
  --val-manifest /path/to/gen1_304x240/manifests/val.jsonl \
  --output-dir /path/to/runs/gen1_detection_smoke \
  --window-ms 40 \
  --epochs 3 \
  --eval-every 1 \
  --max-train-frames 2000 \
  --max-val-frames 1000
```

smoke test完走後はframe上限を外します。`train.jsonl`にYOLOX lossと`AP/AP_50/AP_75/AP_S/AP_M/AP_L`、`checkpoint-latest.pt`に再開可能なhead・optimizer状態を保存します。

## Window sweepの集計

下流evaluatorが次のJSONLを出力するものとします。

```json
{"method":"window_jepa","metric":"mAP","seed":0,"window_ms":5,"value":0.311,"higher_is_better":true,"sample_set_id":"<anchor SHA-256>","window_group":"unseen_extrapolation"}
{"method":"window_jepa","metric":"mAP","seed":0,"window_ms":40,"value":0.402,"higher_is_better":true,"sample_set_id":"<anchor SHA-256>","window_group":"seen"}
```

```bash
window-jepa-summarize \
  --input outputs/gen1/window_metrics.jsonl \
  --reference-window-ms 40 \
  --minimum-seeds 3
```

metric方向は各行の`higher_is_better`で宣言するため、mAPとEPEを取り違えません。全手法・seedでwindow集合、window group、anchor集合hashが一致しない入力は拒否します。集計値は全窓・seen・未見補間・未見外挿ごとの平均、worst、40 msからの最大悪化、相対悪化、`log(Δ)`上の台形積分平均と、3 seed以上の平均・標準偏差です。

## 因果性と再現性

- 抽出区間は一貫して`(start, end]`です。開始・終了の双方を`searchsorted(..., side="right")`で処理します。
- contextとtargetは同じ`sequence_id`、同じ`t_end_us`を共有します。
- anchorはイベントindexでなくsequenceの時間軸から一様に選びます。
- mask、window pair、crop、flipは`seed + epoch + sample index`から決定的に生成され、worker数に依存しません。
- 最大窓が収まるanchor集合を先に定義するため、評価するΔごとにsample集合が変わりません。
- サンプル総イベント数による正規化は行いません。`log1p`は各voxel cellへ要素ごとに適用します。

## テスト

テストには境界timestampの重複、因果性、時間bin補間、空窓、共有transform、window ratio、mask非重複、target leakage、EMA式、canonical shape、metric方向を含めています。

```bash
pytest
```

この作業環境では依存関係を伴うテストを実行せず、Python標準ライブラリによる構文検証のみを行う方針です。

## 実験を進める順序

1. 40 ms固定の下流モデルを全評価窓でsweepし、問題が実在するか確認
2. random-window augmentation baseline
3. `direct_consistency.yaml`による直接feature一致
4. 40 ms固定targetの最小Window-JEPA
5. random target・patch masking・canonical latent
6. Gen1/DSEC/MVSECの2データセット・2タスク以上で比較

主KPIは全窓平均だけでなく、未学習窓でのworst性能と40 msからの最大悪化です。補間窓（15/30/60 ms）と外挿窓（5/120 ms）は分けて報告してください。

## 参考実装とライセンス境界

設計確認には`tmp/ijepa`、`tmp/vjepa2`、`tmp/RVT`、`tmp/fast-feature-fields`を使用しました。本体コードは独自実装で、参照リポジトリからコードをコピーしていません。詳細は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
