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
元時計、camera、歪み座標系、source/stored解像度、整数downsample倍率も各行に保持するため、
下流ラベルを同じ時計・座標へ明示的に変換できます。

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

学習中はrank 0だけにepoch単位の進捗バーを表示し、loss、prediction/target std、
learning rateだけを簡潔に更新します。JSONLの完全な記録は従来どおり
`OUTPUT_DIR/train.jsonl`へ保存し、TensorBoardには次の最小6系列だけを書きます。

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
