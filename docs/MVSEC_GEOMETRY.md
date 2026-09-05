# MVSECでのJEPA + CMax学習と幾何評価

## 実装の位置づけ

この経路は、MVSECの`outdoor_day2`でevent-only事前学習を行い、同じcheckpointを
optical flowとmetric depthで評価するStage 1です。比較の中心は次の2条件です。

1. `recurrent future-JEPA`
2. `recurrent future-JEPA + CMax`

[Fast Feature Field (F³)](https://arxiv.org/html/2509.25146)から借りるのは、MVSECの
recording、raw/distorted座標、flow mask、depth範囲などの実験契約です。F³本体を再実装した
ものではありません。F³はpast eventからfuture event occupancyを予測するhash-field表現であり、
そのunsupervised flow headはGaussian pyramid上のfeature constancyとsmoothnessで学習します。
本projectはJEPA latent predictionを主目的とし、raw eventのCMaxを補助目的として使います。

実装済みの入口は次のとおりです。

| 目的 | 入口 |
| --- | --- |
| 公式HDF5/flow NPZの安全な取得 | `scripts/download/download_mvsec.py` |
| canonical EventStoreへの変換 | `scripts/preprocess/preprocess_mvsec.sh` |
| JEPA / JEPA+CMax事前学習 | `configs/pretrain/*_mvsec*.yaml` |
| CMax headのGT flow評価 | `window-jepa-mvsec-flow cmax-eval` |
| frozen encoder + flow probe | `window-jepa-mvsec-flow probe` |
| frozen encoder + depth probe | `window-jepa-mvsec-depth-probe` |
| 複数条件・seedの安全な実行 | `scripts/experiments/{train,eval}_mvsec_ablation.sh` |
| sample・metric・CMaxの可視化 | `scripts/experiments/visualize_mvsec_ablation.sh` |

多数の条件を比較する場合の推奨順序、day2 label-holdout dev、day1 sealed final、seedの階層、
可視化artifactは[MVSEC ablationガイド](MVSEC_ABLATIONS.md)にまとめています。

## 固定したデータ契約

- sensorはDAVIS346のnative `346×260`、イベントとGTはleftのraw/distorted座標です。
- 前処理ではrectify、resize、cropを行いません。左・右cameraのsource時計を別々に保持し、
  原本HDF5はread-onlyでchunk読込します。
- ViT-S/16へ入れるときだけ、全視野を失わないよう中央へzero-padして`352×272`にします。
  padding領域はイベントなし・GT無効です。
- `outdoor_day2`のleft+rightを事前学習に使います。左右は同じrecording splitに置き、
  right cameraの壊れたgrayscale imageは使用しません。
- `outdoor_day1`はfinal test専用です。flowは内部時刻
  `222.4 s <= t < 240.4 s`を時刻で選択し、checkpoint選択には使いません。
- `outdoor_night1`は任意のdepth OOD testです。
- GT flowはMVSECがdepthとposeから生成した主にrigid ego-motionのflowです。
  `observed flow - rigid flow`による独立運動物体の評価は、このStage 1には含めません。

F³の実装はleft/rightを別々に0起点化した後、単一の`absolute_start_time`属性をright側で
上書きします。また、同じ走行のleft/rightをtrain/validationへ分ける設定があります。本実装は
これらを移植せず、camera別origin、整数microsecond、recording単位splitをmanifestへ保存します。

## 1. Download

まずネット接続も書込みも行わない計画表示で、対象と必要容量を確認します。

```bash
python scripts/download/download_mvsec.py \
  --root /datasets/downloads/mvsec \
  --profile stage1 \
  --plan-only
```

`stage1`はday1/day2の`*_data.hdf5`、depth/pose用`*_gt.hdf5`、公式
`*_gt_flow_dist.npz`で、正確には`102,646,291,553 bytes`（約95.597 GiB）です。
night OODを含む`stage1-ood`は`122,227,194,379 bytes`（約113.833 GiB）です
（night1はdepth-onlyでflow NPZを含みません）。calibrationはStage 1のraw座標評価には
不要ですが、将来rectificationやstereoを行う場合は`--include-calibration`を加えます。

実取得時だけ`gdown`が必要です。

```bash
python -m pip install -e '.[download]'

python scripts/download/download_mvsec.py \
  --root /datasets/downloads/mvsec \
  --profile stage1
```

転送は固定したGoogle Drive file IDから行い、未完了データを`.part`として残して再開します。
開始前に残容量+1 GiBを確認し、完了時に正確なbyte数、HDF5 signature/schema、flow NPZの
`timestamps,x_flow_dist,y_flow_dist` key・shape・NPY header・member CRCを検査してからatomic renameします。
配布元は暗号学的checksumを公開していないため、生成するSHA-256 sidecarは初回検証後の
ローカルcache同一性を守るもので、publisher真正性の証明ではありません。

## 2. Canonical前処理

Google Drive GUIから取得した場合、`indoor_flying/`、`outdoor_day/`、`outdoor_night/`が直下に
あるdirectoryを`--raw-root`へそのまま渡せます。downloader形式のcontainer root
（その下に`raw/outdoor_day/`がある形）も自動判定します。巨大HDF5の移動や複製は不要です。

ただし公式HDF5にはdense flowが含まれません。別配布の
`outdoor_day1_gt_flow_dist.npz`（7,389,716,086 bytes）と
`outdoor_day2_gt_flow_dist.npz`（17,555,972,270 bytes）を取得し、それぞれのdata/GT HDF5と
同じ`outdoor_day/`へ置いてください。トップ階層のGoogle Drive分割ZIP、`indoor_flying/`、
calibration ZIPはStage 1 wrapperの対象外です。

```bash
python -m pip install -e '.[hdf5]'

bash scripts/preprocess/preprocess_mvsec.sh \
  --python-bin python \
  --raw-root /datasets/downloads/mvsec/raw \
  --bundle-root /datasets/evjepa/mvsec \
  --plan-only
```

計画が正しければ`--plan-only`を外します。night1も変換する場合は`--include-night`を付けます。

```bash
bash scripts/preprocess/preprocess_mvsec.sh \
  --python-bin python \
  --raw-root /datasets/downloads/mvsec/raw \
  --bundle-root /datasets/evjepa/mvsec \
  --include-night
```

出力は次の構成です。

```text
/datasets/evjepa/mvsec/
├── events/{train,test,ood_test}/*.h5
└── manifests/{train,test,ood_test}.jsonl
```

manifestはcanonical event HDF5に加え、left camera行だけに元のflow NPZ
（`x_flow_dist`,`y_flow_dist`,`timestamps`）とHDF5の
`/davis/left/depth_image_raw{,_ts}`への相対参照を持ちます。flowのformat、Drive file ID、
正確なbyte数、ローカルSHA-256、metadata versionも固定します。HDF5内の代替flowは自動採用せず、
`--mvsec-flow embedded-hdf5`を明示した場合だけ使用します。
depth/pose GT HDF5も、download sidecarが現在のfile ID・byte数・mtimeと一致する場合はSHA-256を
manifestへ伝播し、不一致またはsidecar不在なら現在のsize/mtimeを保存して下流で再検証します。
2018-09-26より古い`outdoor_day1_gt`にはdepth timestampが0.225 sずれる既知問題があります。
一定offsetはschema/range検査では検出できません。このdownloaderが固定する現行公式Drive file IDと
正確なbyte数を使い、ローカルSHA-256を記録してください。別経路の取得物はevent/depthの可視alignmentも
確認します。

rawラベルはbundleへコピーされません。manifestの相対参照を切らないよう、前処理後も元の
data/GT HDF5とflow NPZを削除・移動しないでください。

## 3. JEPAとJEPA+CMax

次の2設定は、CMax sectionと出力先以外を同じにしています。いずれも50 msの非重複窓、
ConvLSTM、future latent prediction、Frame SIGRegを使い、今回の比較を濁らせないため
Temporal SIGReg、window-level Rate Alignment（RA）、Latent Straightening（LS）は0です。

- `configs/pretrain/recurrent_future_convlstm_vits_mvsec.yaml`
- `configs/pretrain/recurrent_future_convlstm_vits_mvsec_cmax.yaml`

各configの`data.manifest`を実際の`manifests/train.jsonl`へ合わせてから実行します。

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=1 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/recurrent_future_convlstm_vits_mvsec.yaml \
  --milestone-epochs 10 25 50 75 100

PYTHONPATH=src torchrun --standalone --nproc-per-node=1 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/recurrent_future_convlstm_vits_mvsec_cmax.yaml \
  --milestone-epochs 10 25 50 75 100
```

CMax設定のevent上限1,024、loss weight 0.05、最大変位32 px/50 msは最初のmemory-safeな
仮説値です。まず`cmax/valid_window_fraction`、flow saturation、zero/shuffled-flow対照、
JEPA lossとlatent collapse指標を確認し、同時に改善しないrunは採用しません。

RA/LS、Temporal SIGReg、CMax weight、context、reference/scaleを一軸ずつ変えるimmutable configと
複数seed runnerは[MVSEC ablationガイド](MVSEC_ABLATIONS.md)にあります。RA/LSはNeural Eventsの
event-level logit lossそのものではなく、recurrent patch tokenへ適用したwindow-level adaptationです。

## 4. Optical flow

以下の直接commandは、条件をday2 devで選び終えた後のfinal評価例です。多数のcheckpointをここへ
投入せず、条件選択にはablation runnerの`--stage dev`を使ってください。

### CMax headをそのまま評価

```bash
window-jepa-mvsec-flow cmax-eval \
  --checkpoint /checkpoints/mvsec-cmax/checkpoint-epoch0100.pt \
  --eval-manifest /datasets/evjepa/mvsec/manifests/test.jsonl \
  --output-dir outputs/mvsec-flow/cmax-direct \
  --protocol-stage final \
  --alignment causal \
  --dt native
```

これはCMax学習済みheadのzero-shot物理精度です。低いCMax lossだけでは正しいflowを保証しないため、
必ずこのGT評価と併記します。

### Frozen encoder probe

```bash
window-jepa-mvsec-flow probe \
  --checkpoint /checkpoints/mvsec-jepa/checkpoint-epoch0100.pt \
  --train-manifest /datasets/evjepa/mvsec/manifests/train.jsonl \
  --eval-manifest /datasets/evjepa/mvsec/manifests/test.jsonl \
  --output-dir outputs/mvsec-flow/jepa-random-head \
  --head-init random \
  --protocol-stage final \
  --alignment causal \
  --dt native \
  --epochs 30
```

JEPA-onlyとJEPA+CMaxの表現比較では、両方とも`--head-init random`、同じseed、同じepoch数を使います。
CMax headのarchitecture ablationとencoder品質が交絡しないよう、random probeはcheckpointのCMax
設定から独立した固定仕様です。
CMax headをwarm startする`--head-init cmax`は別の診断であり、encoder品質とhead事前学習の効果を
分離できないため主比較には使いません。このprobeはday2のGT EPEでheadのみを学習する独自の
supervised probeであり、F³のunsupervised feature-constancy headの再現値ではありません。

評価はfiniteかつnonzero GT、event-supportがあるpixel、元sensorの上193 rowだけを使い、
AEPE、1PE/2PE/3PE、AAEを出します。event-support区間はprotocolごとに異なり、
`causal+native`は各reference直前の実flow interval、`causal+dt1`はlabel直前22,222 µs、
`f3_centered`はlabelを中心に置く最終50 ms context全体です。F³互換のsample平均と、全valid pixelで
重み付けしたglobal平均を両方保存します。選択したGT indexと内部timestampのSHA-256もreportへ
記録します。

- `--alignment causal`はGT timestampまでのeventだけを使用する主結果です。
- `--alignment f3_centered`はGT前後のeventを含む再現診断で、reportにfuture-event useを明記します。
- `--dt native`は公式NPZのnative timestamp間隔とflowをそのまま使います。
- `--dt dt1`はnative flowを22,222 µsへscalar換算する診断で、45 Hzへの時間resamplingでは
  ありません。したがってexact EV-FlowNet 800-frame再現値とは呼びません。
- `--dt dt4`は複数区間flowの正しい合成を未実装のため、誤って単純倍率で評価せず明示エラーにします。

exact EV-FlowNet 800-frame protocolにはAPS timestampsに合わせたpose/depth補間またはflow合成が
必要で、現時点では未実装です。

## 5. Metric depth

```bash
window-jepa-mvsec-depth-probe \
  --checkpoint /checkpoints/mvsec-jepa/checkpoint-epoch0100.pt \
  --train-manifest /datasets/evjepa/mvsec/manifests/train.jsonl \
  --eval-manifest /datasets/evjepa/mvsec/manifests/test.jsonl \
                  /datasets/evjepa/mvsec/manifests/ood_test.jsonl \
  --output-dir outputs/mvsec-depth/jepa \
  --protocol-stage final \
  --alignment causal \
  --epochs 30 \
  --device cuda \
  --precision bf16
```

frozen recurrent encoderの最終patch tokenから小型log-depth headだけを学習します。day2で固定epoch数を
学習し、day1/night1は最終評価にだけ使い、test metricによるbest checkpoint選択はしません。
raw/distorted `depth_image_raw`を直接metric depthとして扱い、sentinel disparityへの往復変換は
行いません。valid範囲はfiniteかつ`0.1 m < depth < 80 m`です。

公式downloader経由なら整合する`.verified.json`のSHA-256を再利用して大容量GTの再hashを避けます。
sidecarがない、または現在のsize/mtimeと一致しない場合は、再現性記録のためprobe初回にGT HDF5全体の
SHA-256をstream計算するため時間がかかります。

δ1/δ2/δ3、AbsRel、標準SqRel、RMSE、RMSE-log、log10、標準SILogと、F³実装固有の
`F3_SqRel`・`F3_SILog`をsample平均とpixel平均で保存します。10/20/30 m未満のMAEも併記します。

## 推奨する最小実験表

| Encoder pretrain | Flow head | 目的 |
| --- | --- | --- |
| JEPA | random probe | 表現baseline |
| JEPA+CMax | random probe | CMaxによるencoder改善の検証 |
| JEPA+CMax | checkpoint CMax head | zero-shot flowの物理精度 |
| JEPA+CMax | CMax-init probe | head warm startの参考値 |
| JEPA+RA | random probe | event-rate整合の寄与 |
| JEPA+LS | random probe | latent方向整合の寄与 |
| JEPA+RA+LS | random probe | 両時間正則化の相互作用 |
| JEPA+RA+LS+CMax | random probe | latent dynamicsと幾何補助目的の相互作用 |

各条件を少なくとも3 seedで比較し、flowだけでなくdepth、JEPA loss、latent rank/std、CMaxの
zero/shuffled対照を同時に確認します。Stage 1の成功条件は「CMax lossが下がる」ことではなく、
同じprobe契約でJEPA-onlyよりheld-out flow/depthまたは別下流性能が改善し、collapseが悪化しないことです。

## 現時点で含めないもの

- F³ hash-field backboneやfuture-event focal objectiveの再実装
- F³のGaussian-pyramid feature-constancy flow trainingそのもの
- `dt=4`のmulti-interval flow composition
- depth + ego-motionからの`F_rigid(D,T)`再生成
- `F_observed - F_rigid`によるdynamic-object segmentation
- 実データ全量での数値再現保証

次段階では、まずこのStage 1でCMax追加の寄与を固定し、その後にdepth/poseからrigid flowを生成して
観測flowとの差を調べます。動的物体の定量評価にはMVSECだけでなくEVIMO2等を追加する必要があります。

## 一次資料

- [Fast Feature Field (F³) paper](https://arxiv.org/html/2509.25146)
- [Fast Feature Fields official implementation](https://github.com/grasp-lyrl/fast-feature-fields)
- [MVSEC overview / license / citation](https://daniilidis-group.github.io/mvsec/)
- [MVSEC HDF5 download and flow information](https://daniilidis-group.github.io/mvsec/download/)
- [MVSEC data format](https://daniilidis-group.github.io/mvsec/data_format/)
- [MVSEC change log](https://daniilidis-group.github.io/mvsec/change_log/)
- [Official MVSEC flow-generation code](https://github.com/daniilidis-group/mvsec/tree/master/tools/gt_flow)
