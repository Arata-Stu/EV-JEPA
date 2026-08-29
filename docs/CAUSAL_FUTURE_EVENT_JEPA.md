# Causal Future Event JEPA

## 目的

この実装は、イベント列の過去から未来の**特徴量**を予測し、画素再構成なしで時間的な
表現を事前学習するためのものです。主目的は未来latent predictionであり、SIGRegはその
学習崩壊を抑える補助項です。

\[
\mathcal L
=\mathcal L_{\mathrm{future}}
+\lambda_f\mathcal L_{\mathrm{frame/support\ SIGReg}}
+\lambda_h\mathcal L_{\mathrm{temporal\ SIGReg}}.
\]

ここでframe/support項は、Supportがglobalで有効な時刻にはFrameとSupportの平均、無効な
時刻にはFrame単独です。

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
  --config configs/pretrain/recurrent_future_convlstm_vits_gen1.yaml
```

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
