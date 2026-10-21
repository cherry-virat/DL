3
const fs = require('fs');
const path = require('path');

const maskBuf = fs.readFileSync(path.resolve(__dirname, '../outputs/api_runs/REQ-0F862D28/masks/REQ-0F862D28_glioma_seg.nii.gz'));
const arrayBuf = maskBuf.buffer.slice(maskBuf.byteOffset, maskBuf.byteOffset + maskBuf.byteLength);

global.window = global;
global.URL = require('url').URL;

const niivue = require(path.resolve(__dirname, '../frontend/niivue.umd.js'));

try {
    const m1 = new niivue.NVImage(arrayBuf, "Segmentation Mask");
    console.log("Mask 1 loaded? imageType:", m1.imageType);
} catch (e) {
    console.log("Mask 1 FAILED:", e.message);
}

try {
    const m2 = new niivue.NVImage(arrayBuf, "segmentation_mask.nii.gz");
    console.log("Mask 2 loaded? imageType:", m2.imageType, "cal_min:", m2.cal_min, "cal_max:", m2.cal_max);
} catch (e) {
    console.log("Mask 2 FAILED:", e.message);
}
