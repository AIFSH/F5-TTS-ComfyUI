# F5-TTS-ComfyUI
a custom node for [F5-TTS](https://github.com/SWivid/F5-TTS),you can find [workflow here](./doc/base_workflow.json)

## Weights
weights will be download from hf automaticlly,对于国内用户，你可以手动下载解压后把F5-TTS文件夹放到`ComfyUI/models/AIFSH`目录下面,[下载地址](https://pan.quark.cn/s/e3a3e4281ada)

## 教程
- [演示视频](https://www.bilibili.com/video/BV1Tjm5YLEsX)
- [一键包,内含F5-TTS，FireRedTTS，JoyHallo，hallo2四个节点，持续更新中，一次订阅31天免费更新](https://b23.tv/Zm3kPNP)

## Example

| gen_text | ref_audio | out_audio | audio_img |
| -- | -- | -- | -- |
|`你好，我是太乙真人！欢迎来四川找我玩`| <video src="https://github.com/user-attachments/assets/6758239e-9215-4301-ba06-ac9dad06c306" /> | <video src="https://github.com/user-attachments/assets/2f08ad54-0728-4542-84d3-6e8588b6ef3d" /> | ![](./doc/ComfyUI_temp_dgtgr_00001_.png) |
|`有的人叫我自然，也有的人尊称我为自然母亲`|  <video src="https://github.com/user-attachments/assets/89fde537-abba-4959-9e8f-03230d76014a" /> | <video src="https://github.com/user-attachments/assets/c4058295-1db1-4009-af7d-4c84339eae59" /> | ![](./doc/ComfyUI_temp_rhsxy_00001_.png)
