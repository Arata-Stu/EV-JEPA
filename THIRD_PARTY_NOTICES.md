# Third-party references

このプロジェクトの本体コードは独自実装です。以下のローカルcloneを設計資料として参照し、コードの直接コピーは行っていません。

| Repository | 参照した設計 | License |
| --- | --- | --- |
| facebookresearch/ijepa | online/EMA encoder、multiblock masking、latent loss | CC BY-NC 4.0 |
| facebookresearch/vjepa2 | predictor、2D ViT、checkpoint設計 | MIT（一部Apache-2.0ファイルあり） |
| uzh-rpg/RVT | event histogram、Prophesee evaluation、label alignment | MIT |
| grasp-lyrl/fast-feature-fields (`320eec6`) | MVSEC split、raw-coordinate flow/depth protocol、feature probe設計 | Apache-2.0 |
| daniilidis-group/mvsec (`ee9b7ac`) | MVSEC HDF5 schema、GT flow生成法、評価規約 | MIT（code） |

データconverterのschema確認には、DSEC公式Data Format、M3ED公式loader、
Prophesee DAT公開仕様／toolbox、hdf5plugin公式Usageを参照しました。converterは
これらのコードをコピーせず独自実装しています。`configs/datasets/`のDSEC sequence名は
DSEC-Detectionが公開する公式論理splitを転記したものです。

MVSECのdataset本体は公式project pageでCC BY-SA 4.0として公開されています。MVSECを使った
成果ではdataset論文を引用し、配布GT optical flowを使う場合は公式案内に従ってEV-FlowNetも
引用してください。MVSEC code repositoryのMIT Licenseをdataset binaryのlicenseと混同しません。

F³のMVSEC processorは、原本を`r+`で開く処理、cameraごとの時刻0起点化、left/rightで単一
`absolute_start_time`属性を上書きする処理を含みます。本projectはそのコードをコピーせず、原本を
read-onlyでchunk読込し、camera別originを保持する独自converterにしています。F³ flowは
feature constancy方式であり、本projectのCMax補助目的やsupervised frozen probeと同一ではありません。

今後、参照実装のファイルをコピーまたは改変して取り込む場合は、元著作権表示・ライセンス本文・変更内容をこのファイルへ追記してください。特にI-JEPAコードの直接利用には非商用条件があるため、本体へ取り込まない方針です。

`event_window_jepa.downstream.gen1_detection`は、利用者が別途用意したRVT checkoutから
MITライセンスのYOLOX head、postprocess、Prophesee evaluatorを実行時にimportします。
RVTコード自体はこのリポジトリへコピーしません。利用時はRVTのLICENSEと、RVTが
改変元として明記するYOLOXのライセンス・著作権表示にも従ってください。
