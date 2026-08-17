# Third-party references

このプロジェクトの本体コードは独自実装です。以下のローカルcloneを設計資料として参照し、コードの直接コピーは行っていません。

| Repository | 参照した設計 | License |
| --- | --- | --- |
| facebookresearch/ijepa | online/EMA encoder、multiblock masking、latent loss | CC BY-NC 4.0 |
| facebookresearch/vjepa2 | predictor、2D ViT、checkpoint設計 | MIT（一部Apache-2.0ファイルあり） |
| uzh-rpg/RVT | event histogram、Prophesee evaluation、label alignment | MIT |
| grasp-lyrl/fast-feature-fields | causal raw-event slicing、flow/segmentation probe | Apache-2.0 |

データconverterのschema確認には、DSEC公式Data Format、M3ED公式loader、
Prophesee DAT公開仕様／toolbox、hdf5plugin公式Usageを参照しました。converterは
これらのコードをコピーせず独自実装しています。`configs/datasets/`のDSEC sequence名は
DSEC-Detectionが公開する公式論理splitを転記したものです。

今後、参照実装のファイルをコピーまたは改変して取り込む場合は、元著作権表示・ライセンス本文・変更内容をこのファイルへ追記してください。特にI-JEPAコードの直接利用には非商用条件があるため、本体へ取り込まない方針です。
