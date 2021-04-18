This is C++ version of [FairMOT](https://github.com/ifzhang/FairMOT) using TensorRT in Windows.

The detection and feature extraction parts using TensorRT can refer to [tensorRTIntegrate](https://github.com/dlunion/tensorRTIntegrate).


## How to Run
1. export onnx model file in [FairMOT](https://github.com/ifzhang/FairMOT) by adding the following code in line 479 in "/src/lib/models/networks/pose_dla_dcn.py"
```
	z = {}
	for head in self.heads:
		z[head] = self.__getattr__(head)(y[-1])

	hm = z["hm"]
	wh = z["wh"]
	reg = z["reg"]
	hm = F.sigmoid(hm)
	hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)


	id_feature = z['id']
	id_feature = F.normalize(id_feature, dim=1)
	id_feature = id_feature.permute(0, 2, 3, 1).contiguous() #switch id dim
	return [hm, wh, reg, hm_pool, id_feature]
```
2. Follow [tensorRTIntegrate](https://github.com/dlunion/tensorRTIntegrate) to build TensorRT engine which has been covered in this repository. You need to follow [tensorRTIntegrate](https://github.com/dlunion/tensorRTIntegrate) first to make sure DCN works and then add the MOT part in the folder "mot".
## Acknowledgement

- Kalman Filter is borrowed from [DeepSort](https://github.com/bitzy/DeepSort), [deep_sort] https://github.com/apennisi/deep_sort
- [FairMOT]（https://github.com/ifzhang/FairMOT）
- [tensorRTIntegrate] (https://github.com/dlunion/tensorRTIntegrate)