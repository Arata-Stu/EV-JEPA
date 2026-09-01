# Causal Future Event JEPA

## 目的

この実装は、イベント列の過去から未来の**特徴量**を予測し、画素再構成なしで時間的な
表現を事前学習するためのものです。主目的は未来latent predictionであり、SIGRegはその
学習崩壊を抑える補助項です。

\[
\mathcal L
=\mathcal L_{\mathrm{future}}
+\lambda_f\mathcal L_{\mathrm{frame/support\ SIGReg}}
+\lambda_h\mathcal L_{\mathrm{temporal\ SIGReg}}
+\lambda_c\mathcal L_{\mathrm{CMax}}.
\]

ここでframe/support項は、Supportがglobalで有効な時刻にはFrameとSupportの平均、無効な
時刻にはFrame単独です。CMaxは既定OFFのraw-event motion補助目的であり、SIGRegの代替では
ありません。設計、公式Taming実装との関係、設定制約、成功判定は
[CMax補助目的の設計と評価契約](CMAX_AUXILIARY_OBJECTIVE.md)を参照してください。

既存の`recurrent_window_jepa`を置換せず、独立した
`recurrent_future_jepa` objectiveとして追加しています。旧checkpointと旧実験の意味は
維持されます。

## 計算グラフ

```text
online (causal)
E_t ── Frame ViT Fθ ── f_t ── ConvLSTM(h_{t-1}) ── h_t
  │                         │                         │
  │                         └─ Frame SIGReg          ├─ Predictor ── ẑ_{t+k}
  │                                                   └─ Temporal SIGReg
  │
target (stateless, stop-gradient)
E_{t+k} ── EMA Frame ViT Fξ ─────────────────────────────────────── z_{t+k}
```

- `Fθ`は各イベント窓を独立に符号化するfull-frame V-JEPA 2.1型ViTです。
- ConvLSTMはViTの**後段**にあり、frame token gridを時間方向へ統合します。
- `Fξ`は`Fθ`のEMA copyですが、target forwardではrecurrent cellを完全に迂回します。
- EMA targetに時間stateはなく、未来窓以外の情報は入りません。
- onlineとtargetでframe ViTの入力分布を揃えるため、どちらもfull/unmaskedです。

実装上はEMA parameter名とcheckpoint互換性を保つため、target objectにも未使用のrecurrent
cellが存在します。ただし`forward_frame()`を使うので、target latentはそのparameterを変更しても
不変です。

## maskの契約

`recurrent_future_jepa`では、datasetが生成したrandom spatial maskを学習グラフへ渡しません。

| 箇所 | mask |
|---|---|
| online Frame ViT | なし |
| burn-inを含むConvLSTM更新 | なし |
| EMA Frame ViT | なし |
| Predictorのcontext position | all-true |
| prediction loss | event supportでactive/inactiveを均衡化 |

したがって、maskがGRU/LSTMへ入り、欠損patternそのものを時間stateが記憶する経路はありません。
旧objectiveとの共通dataset contractを保つためmask自体は生成されますが、future objectiveでは
logging用の診断値です。

## 未来窓のアラインメント

設定`prediction_horizon_steps: k`に対し、datasetは一度だけ
`online_steps + k`窓をvoxel化します。その後、

```text
x[i]        = sampled[i]
x_future[i] = sampled[i + k]
```

を同じ長さで返します。標準設定`k=1`は次の11窓です。

```text
sampled:  [burn-in 2] [loss-bearing context 8] [lookahead 1]
online x: [burn-in 2] [loss-bearing context 8]
future:   [shift後burn-in 2] [shift後target 8]
```

lookaheadはteacher-onlyで、現在batchのonline stateを更新しません。stream samplingの進行幅は
11ではなくonline側の10窓です。これにより前chunkのlookaheadが次chunkの最初のonline窓と
一致し、時間を1窓飛ばす問題を避けています。sequence末尾ではlookaheadを確保できるchunkだけを
sampleします。

## Event-Support balanced prediction loss

EMA teacherは全patchのtarget latentを先に計算します。その後、未来窓のraw event countから
active/inactiveを分け、sampleごとに

\[
L_b=
\begin{cases}
\frac12(L_{b,\mathrm{active}}+L_{b,\mathrm{inactive}}), & \text{両方あり},\\
L_{b,\mathrm{active}}, & \text{activeのみ},\\
L_{b,\mathrm{inactive}}, & \text{inactiveのみ}
\end{cases}
\]

を計算し、最後にsample平均します。event-rich sampleがactive patch数だけで他sampleより重くなる
ことはありません。片方の集合が空でもNaNを出しません。

## Frame／Temporal SIGReg

各時刻・各sampleから1本だけvectorを作ります。active patchは重み1、inactive patchは
`activity_floor`でpoolingします。event数そのものをbatch weightにしないため、sequence間の
寄与は均等です。

- Frame SIGReg: `f_t`のevent-support pooled vector
- Support SIGReg: active patch平均とinactive patch平均の差
- Temporal SIGReg: `(h_t - h_{t-1})`のevent-support pooled vector

それぞれ独立したFP32 MLP projectorを通し、random 1-D projection上の経験特性関数を標準正規分布
へ合わせるsliced Epps–Pulley lossを計算します。projection方向とprojector初期値は明示seedで
全rank共通です。各時刻に`[B,D]`で呼び、`B×T`へflattenしません。zero-support rankでも
distributed collectiveの回数と順序は変えません。

FrameとSupportの両方が有効な時刻では2項を平均し、Supportのglobal有効sampleが2未満なら
Frame項をそのまま使います。これにより正則化scaleを保ちながら、全patchがsample-globalな
同一vectorへ崩れる空間collapseも検出します。`support_sigreg_samples`が長時間2未満なら、
`active_min_events`または窓・patch設定を見直します。thresholdにより全patchが同じclassに
なってもraw event countに差があれば、低count側と高count側のcontrastを使います。全patchの
countまで同値なら、位置indexだけに依存した分割を避けるためSupport項はそのsampleを除外します。

## 実行設定

V100 32 GB × 3向けの初期設定:

```bash
torchrun --standalone --nproc-per-node=3 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/recurrent_future_convlstm_vits_gen1.yaml \
  --milestone-epochs 10 25 50 75 100
```

ConvLSTMの寄与を測るFrame-only baselineは、同じsequence clip、未来alignment、EMA teacher、
balanced event-support loss、Frame/Support SIGRegを使い、online側だけを
`Frame ViT → Predictor`とします。Temporal SIGRegは使いません。

```bash
torchrun --standalone --nproc-per-node=3 \
  -m event_window_jepa.train.pretrain \
  --config configs/pretrain/frame_future_vits_gen1.yaml \
  --milestone-epochs 10 25 50 75 100
```

Frame-onlyでもsample matchingのためrecurrent runと同じburn-in prefixをdatasetから読みますが、
loss対象外のprefixはencoderへ通しません。`recurrent_state_rms`とTemporal SIGReg診断は0／対象外で、
それ以外のfuture loss・collapse診断は同じ名前で記録されます。

標準設定はper-rank 8本、3 GPUで合計24本のstream laneを使うため、manifestには少なくとも
24個の異なる`source_recording_id`が必要です。開始時に不足を検出した場合は、manifestを確認するか、
最初のrandom-clip成立確認だけ`recurrent.sampling: random`へ切り替えてください。

future objectiveは既定で少なくとも一方のSIGReg weightを必須にします。正則化なしの対照実験だけは
`future_prediction.allow_unregularized: true`を明示してください。設定漏れでcollapse対策なしの
正式runを開始することはありません。

初回の正式学習前には、同じGPU数・batch size・precisionで短いsmoke runを作り、少なくとも
次を確認してください。

- `future_prediction_loss`が減少する
- `prediction_std`と`target_std`が0へ単調に落ちない
- `sigreg_to_prediction_ratio`が急増しない
- `frame_sigreg_samples`、`support_sigreg_samples`、`temporal_sigreg_samples`が有効
- active/inactive prediction lossの片方だけが発散しない
- `recurrent_state_rms`が有限
- FP16 overflowによる連続skipがない

標準設定のSIGReg slice数は計算量を抑えた256です。1024 slicesは安定学習を確認した後の
ablationとして扱います。SIGReg weightの比較では、Frame/Temporalの有無以外のsampling、EMA、
batch、seedを固定してください。

## checkpointと下流利用

checkpointはonline/EMA encoder、predictor、scale embeddingに加え、Frame／Temporal projectorと
SIGReg bufferも保存します。厳密resumeではこれらが欠けていればerrorになります。

下流では従来どおり`encode_recurrent()`またはstateful detection経路を使います。future設定から
復元したencoderは`recurrent_placement=post_encoder`なので、推論も
`Frame ViT → ConvLSTM`の順序を維持します。Linear Probeでframe-only表現を評価したい場合は
`online_encoder.forward_frame()`を使い、時間表現の評価とは分けて報告してください。

## 特徴量の定性可視化

学習済みcheckpointについて、現在・未来のeventとpatch latentを同じレポートで比較できます。
PCAは評価対象を見て都度fitせず、指定したsample群の**EMA future targetだけ**から1回fitし、
Frame ViT、ConvLSTM、prediction、controlへ同じ基底と表示scaleを適用します。そのため、同一
checkpoint内ではpanel間のRGB差をlatent差として比較できます。別run間ではlatent軸が任意回転
し得るため、RGBそのものではなく数値診断を主に比較してください。またsample indexは各runの
runtime seedから実clipへ写像されるため、run間の数値比較ではJSONの`sample_set_id`が一致する
場合だけsample-matchedとして扱ってください。IDにはanchor、crop、flipも含まれます。

```bash
window-jepa-visualize-future \
  --checkpoint outputs/pretrain/recurrent_future_convlstm_vits_gen1_seed0/checkpoint-latest.pt \
  --calibration-samples 4 \
  --sample-index 0 \
  --output outputs/feature-vis/epoch-100.html
```

checkpointに保存されたmanifestの絶対pathが評価serverと異なる場合だけ、`--manifest`で差し替え
ます。representation、crop、window、stride、horizonなどはcheckpoint設定から固定されます。

```bash
window-jepa-visualize-future \
  --checkpoint /path/to/checkpoint-epoch0100.pt \
  --manifest /path/to/gen1_train_manifest.jsonl \
  --device cuda \
  --output outputs/feature-vis/epoch-100.html
```

既定では、事前に固定した連続4 sampleをPCA calibrationと数値集計に使い、最初のsampleの
最終supervised stepだけを画像化します。全supervised stepを画像化する場合は`--all-steps`を
追加してください。無関係targetを別clipの同じonline stepから選ぶため、calibration sample数は
2以上が必須です。HTMLと同じ場所に完全な数値を含むJSON、`*_assets/`にPNGが保存されます。
別clip履歴の対応を再現・変更する場合は`--history-replacement-seed`を指定します。対応表はJSONの
`history_replacement_clip_permutation`へ保存されます。

`correct`、`history shuffled`、`history reversed`、`history replaced`、`state reset`は同じ現在窓と
正しいEMA future targetを共有します。`history replaced`は別clipの同じ長さのprefixからstateを
作るため、resetに含まれる立ち上がり状態の分布差と、履歴内容の寄与を切り分けられます。
`unrelated target`だけはpredictionを固定し、比較先を別clipの同じonline stepへ交換します。

| 条件 | 変更するもの | 主に確認すること |
|---|---|---|
| correct | なし | 基準となる未来予測 |
| history shuffled | 現在窓を固定し、過去prefixだけ並べ替え | 時系列順序を利用しているか |
| history reversed | 現在窓を固定し、過去prefixを完全な逆順で再生 | 動きの向き・因果順序を利用しているか |
| history replaced | 現在窓を固定し、同じ長さの過去prefixだけ別clipへ交換 | scene固有の履歴内容を利用しているか |
| state reset | 現在窓の直前でstateを初期化 | 過去そのものを利用しているか |
| unrelated target | predictionを固定し、比較先だけ別anchorへ交換 | 時刻・sample固有の未来を予測しているか |

`correct`より各controlのcosine errorが大きければ、その差は「順序」「履歴内容」「stateの継続」
「正しい未来対応」が役立つ証拠になります。未学習modelでも成立する大小関係ではないため、CLIは
合否を決めず、paired
penaltyと固定位置centered effective rankを記録します。error heatmapはpanelごとのmin-maxを行わず、
常にtoken LayerNorm後のcosine error `[0, 2]`を共通scaleで描画します。collapse表はraw latentと
token LayerNorm後を分け、さらに同じonline stepのclip同士でstd/rankを計算します。これにより、
時刻ごとに定数が違うだけの表現をsample間の有効な分散として数えることを避けます。
