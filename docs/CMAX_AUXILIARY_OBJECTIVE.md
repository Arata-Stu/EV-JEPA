# CMax補助目的の設計と評価契約

## 位置づけ

CMax（contrast maximization）は、`recurrent_future_jepa`の主目的を置き換えず、online側の
因果表現にevent motionを追加で学習させる補助目的です。

\[
\mathcal L
=\mathcal L_{\mathrm{future\ latent}}
+\lambda_{\mathrm{SIGReg}}\mathcal L_{\mathrm{SIGReg}}
+\lambda_{\mathrm{CMax}}
 \left(\mathcal L_{\mathrm{CMax}}
 +\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}\right).
\]

この3項は役割が異なります。

- future latent predictionは、過去から未来の特徴を予測できる表現を学習させます。
- SIGRegは、online/target表現が定数へ崩れる**representation collapse**を抑えます。
- CMaxは、raw eventの時刻・位置と予測flowを使い、動きに整合する表現を促します。

CMaxだけでも、eventを少数pixelへ不自然に集める**event collapse**が起こり得ます。そのため
CMaxをSIGRegの代替にはせず、flow上限、smoothness、zero/shuffled-flow対照、IWEの占有範囲を
別々に監視します。

## 一次資料とライセンス

設計上の一次資料は、Paredes-VallésらのICCV 2023論文
[Taming Contrast Maximization for Learning Sequential, Low-latency Event-based Optical Flow](https://openaccess.thecvf.com/content/ICCV2023/html/Paredes-Valles_Taming_Contrast_Maximization_for_Learning_Sequential_Low-latency_Event-based_Optical_Flow_ICCV_2023_paper.html)
と[公式実装](https://github.com/tudelft/taming_event_flow)です。公式実装は
[MIT License](https://github.com/tudelft/taming_event_flow/blob/master/LICENSE)で公開されています。
コードを転載・改変して配布する場合は、MIT Licenseのcopyright noticeとpermission noticeを
保持します。

公式手法の重要な要素は、短いevent partitionをstateful recurrent modelで逐次処理すること、
反復warpingで線形運動仮定を緩めること、複数の参照時刻と時間スケールでCMaxを計算することです。
公式の[学習設定](https://github.com/tudelft/taming_event_flow/blob/master/configs/train_flow.yml)では
`max_num_grad_events: 10000`、flow scaling、iterative warpingなどが明示され、
[loss実装](https://github.com/tudelft/taming_event_flow/blob/master/loss/flow.py)では両端方向のwarping、
時間スケール平均、任意のflow smoothnessが実装されています。

本repoのCMaxは、その考え方をfuture JEPAへ合わせて再実装する補助目的であり、公式DSEC/MVSEC
pipelineの完全再現ではありません。公式値は10 ms入力・別解像度・別network/loss scaleを前提と
するため、Gen1の50 ms・240×304設定へ数値をそのまま移植しません。また、公式loaderは
gradientを通すevent数を制限しつつ残りも別経路で保持しますが、本repoの
`max_events_per_window`は各sample・各base-windowについて、CMax用IWEへ残すevent自体を
決定的一様subsampleする上限です。したがって、公式の10,000と同じ意味ではありません。

## 計算契約

```text
raw events (x, y, t, p) ───────────────────────────────┐
                                                      │ warp / IWE
voxel E_t → Frame ViT → ConvLSTM state h_t → flow head u_t
                            │                         │
                            └→ future predictor       └→ CMax + smoothness
```

- flow headはonline recurrent stateだけを読みます。EMA targetや`x_future`をhead入力にしません。
- CMaxはvoxel/event imageからeventを逆算せず、event storeが返すraw `(x,y,t,p)`を使います。
- raw eventにはonline入力と同じcropとhorizontal flipを一度だけ適用します。flip時は座標とflowの
  x成分の向きが一致しなければなりません。
- JEPAのrandom spatial maskはFrame ViT、ConvLSTM、CMaxのいずれにも渡しません。
- `min_events`未満の窓はCMax対象外ですが、future predictionとSIGRegには通常どおり使います。
- event数を上限で切る場合は、先頭N件ではなく、決定的な一様subsampleを使って時間順に由来する
  系統的な偏りを避け、seedと採用数を再現可能にします。
- flowの単位は`pixels / base-window`です。標準設定ではbase-windowは50 msです。event時刻差に応じて
  変位を比例配分し、複数窓を跨ぐときは各窓のflowを逐次適用します。
- 現実装のglobal timeは`time_index + t`で、各窓のflowを順番に合成します。この定義は連続かつ
  非重複のbase windowを前提とするため、CMax ON時は`recurrent.stride_ms == recurrent.window_ms`を
  必須とします。gapやoverlapを含む系列は設定読込時に拒否します。
- head出力は次式で拘束します。

\[
u_t=d_{\max}\tanh(s_{\mathrm{flow}}\,r_t),
\]

ここで`r_t`はraw head logits、`flow_scale = s_flow`は初期出力を小さくする倍率、
`max_displacement = d_max`だけが絶対上限です。`flow_scale`をpixel単位の変換係数としては使いません。

## 複数参照時刻と時間スケール

`reference_mode`は、各temporal segment内でeventをwarpする参照端を指定します。

- `past`: segment先頭のみ
- `future`: segment末尾のみ。EMA future targetの意味ではありません。
- `both`: 先頭と末尾のlossを同じ重みで平均

`temporal_scales`の各整数は、supervised sequenceを等分するsegment数です。burn-in窓は含みません。
`sequence_length: 8`と`temporal_scales: [1, 2, 4]`なら、次の3段階を平均します。

| partition数 | segment構成 | `reference_mode: both`の参照数 |
|---:|---|---:|
| 1 | 8窓 × 1 segment | 2 |
| 2 | 4窓 × 2 segments | 4 |
| 4 | 2窓 × 4 segments | 8 |

各partition数は`recurrent.sequence_length`を割り切る必要があります。これにより末尾だけ短いsegmentを
暗黙に作らず、scaleごとのloss weightを比較可能にします。またCMax ON時は
`tbptt_steps == sequence_length`を必須とし、全scaleと全supervised履歴へ同じbackwardで勾配を
通します。

## 設定

CMaxは既定OFFです。`cmax` sectionを持たない旧configは`CMaxConfig()`へ解決され、旧checkpointの
config hashでは既定sectionを除外します。`cmax.enabled: true`が、datasetへpacked raw eventsを
要求する唯一の公開switchです。`recurrent.return_raw_events`のような重複flagは設けません。

| key | 意味 | 制約／既定値 |
|---|---|---|
| `enabled` | CMax head・raw-event経路を有効化 | `false` |
| `weight` | 全CMax補助項の外側の重み | OFF時0、ON時正値 |
| `smoothness_weight` | flow spatial smoothnessの内側の重み | 0以上 |
| `hidden_dim` | flow headの中間次元 | 正整数、256 |
| `head_depth` | flow headの層数 | 正整数、2 |
| `reference_mode` | segment内のwarp参照端 | `past/future/both`、既定`both` |
| `temporal_scales` | supervised sequenceの等分数 | 正の昇順unique整数、既定`[1,2,4]` |
| `min_events` | 1 base-windowをCMaxに使う最小event数 | 正整数、128 |
| `max_events_per_window` | 各sample・各base-windowでCMax IWEへ残すraw event上限 | `null`または`min_events`以上 |
| `flow_scale` | tanh前raw logitsの初期倍率 | 正値、0.01 |
| `max_displacement` | 1 base-windowの最大変位 | 正値、32 pixels |

ON時は次を設定読込時に拒否します。

- `optimization.objective != recurrent_future_jepa`
- recurrent sequence loaderまたはConvLSTM/ConvGRUが無効
- `recurrent_placement != post_encoder`
- `tbptt_steps != sequence_length`
- `recurrent.stride_ms != recurrent.window_ms`
- `temporal_scales`が空、非整数、非昇順、重複、または`sequence_length`を割り切らない
- 非有限値、負のloss weight、無効なevent数／flow上限

raw-event返却は`cmax.enabled`からdataset内部で有効化します。raw fieldが欠ける、時刻順でない、座標が
crop外、offsetとevent数が矛盾する場合は、学習stepでskipせずデータ契約errorにします。

## データセット間の移植

CMax経路はGen1のファイル形式を直接読みません。既存の`EventStore`が返す共通の
`EventWindow(x, y, t_us, polarity, height, width)`を、crop/flip後にpacked化します。そのため、
現在対応しているHDF5/NPZ manifestであれば、Gen1以外でも同じ実装を利用できます。

別データセットでは、まず通常のrecurrent future-JEPA configで解像度、window、stride、event表現を
設定し、その上へ`cmax` sectionを追加します。特に`max_displacement`はpixels/base-windowなので、
センサ解像度またはwindow時間を変えたときに同じ数値を無条件で流用しません。
`max_events_per_window`と`min_events`も、そのデータセットのevent-rate分布を確認して決めます。

## Gen1の初期比較

標準設定は
[recurrent_future_convlstm_vits_gen1_horizon200ms_cmax.yaml](../configs/pretrain/recurrent_future_convlstm_vits_gen1_horizon200ms_cmax.yaml)
です。これは
[200 ms recurrent / no Temporal SIGReg baseline](../configs/pretrain/recurrent_future_convlstm_vits_gen1_horizon200ms_no_temporal_sigreg.yaml)
から、`cmax` sectionと出力先だけを変えています。Frame/Support SIGRegは残し、Temporal SIGRegは0の
ままなので、最初の比較は「CMax補助の有無」を主に測れます。

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=3 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/recurrent_future_convlstm_vits_gen1_horizon200ms_cmax.yaml \
  --milestone-epochs 10 25 50 75 100
```

`weight: 0.05`、`smoothness_weight: 0.001`、event上限1,024はGen1 smoke用の初期値であり、
公式論文がこのrepo条件に保証する値ではありません。CMax weightを変更するときは、seed、sampling、
batch、prediction horizon、SIGReg、event capを固定します。

上限使用時のsupervised CMax event数は、rankごとに最大
`batch_size * sequence_length * max_events_per_window`です。標準設定では
`16 * 8 * 1,024 = 131,072`です。V100 16 GBで最初の実行が厳しい場合は512でsmokeを行い、
peak memoryとstep時間を確認してから1,024へ戻します。2,048以上は計測なしに上げません。

## 成功判定

正式比較の前に、同じ3 GPU・batch・precisionで短いsmokeを行い、以下を全て確認します。

1. `cmax_loss`、smoothness、flow norm、全gradientが有限で、EMA targetへCMax gradientが流れない。
2. `cmax/valid_window_fraction`で、post-crop event数が`min_events`以上の窓の割合を記録し、
   初期目標を80%以上とする。低い場合は
   lossを0埋めして見かけ上安定させず、`min_events`と窓設定を見直す。
3. held-out raw eventsで、学習したflowのCMax lossがzero-flowとsample-shuffled flowの両方より低い。
   学習batchだけのIWE sharpness改善は成功と数えない。
4. flowのいずれかの成分が`max_displacement`の95%以上になるpixel割合を記録し、初期目標を5%未満とする。
   飽和が続く場合は上限を安易に広げず、flow scale、loss weight、時刻正規化を確認する。
5. warped eventのoccupied-pixel fractionがzeroへ落ちず、少数pixelへのevent collapseが起きない。
6. `future_prediction_loss`がCMaxなしbaseline同様に低下し、prediction/targetのfixed-position rankと
   stdが0へ崩れない。CMaxだけ改善して未来予測が悪化するrunは採用しない。
7. 同じseedのCMax OFF/ONを比較し、少なくとも3 seedでLinear ProbeまたはGen1 detectionが改善する。
   CMax lossの低下だけをrepresentation改善の根拠にしない。

80%／5%は実装健全性を早期に見るためのlocal guardrailで、Taming論文の報告thresholdでは
ありません。正式な研究結果では、CMax OFF、CMax ON、zero/shuffled-flow対照の設定と全指標を
併記します。
