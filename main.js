import "./nifti-reader-min.js";
 
function readNIFTI(name, data) {
    var canvas = document.getElementById('myCanvas');
    var canvasOut = document.getElementById('myOutCanvas')
    var slider = document.getElementById('myRange');
    var sliderOut = document.getElementById('myRangeOut');
    var niftiHeader, niftiImage;

    // parse nifti
    if (nifti.isCompressed(data)) {
        data = nifti.decompress(data);
    }

    if (nifti.isNIFTI(data)) {
        niftiHeader = nifti.readHeader(data);
        niftiImage = nifti.readImage(niftiHeader, data);
    }

    // TODO : load model and final dimensions
    const newDimensions = [64, 64, 40];
    // set up slider
    var slices = newDimensions[2];
    
    slider.max = slices - 1;
    slider.value = Math.round(slices / 2);
    sliderOut.max = slices - 1;
    sliderOut.value = Math.round(slices / 2);
    
    // slider.oninput = function() {
    //     drawCanvas(canvas, slider.value, niftiHeader, niftiImage);
    // };

    const dataArray = getDataArray(niftiHeader, niftiImage);
    const cols = niftiHeader.dims[1];
    const rows = niftiHeader.dims[2];
    const slices2 = niftiHeader.dims[3];

    // Reshape the data array to 3D
    var data = convertNiftiTo3DArray(niftiHeader, dataArray)
    console.log("Debug by yash starts here");
    console.log(data)
    console.log("The array: ", Array.isArray(data));
    console.log("One elemnet: ", data[1][2][5])
    console.log("Shape of the resulting array:", [data.length, data[0].length, data[0][0].length]);
    console.log(niftiHeader);
    
    var resizedImageData = resizeImageData(data, newDimensions);
    resizedImageData = doNormalization(resizedImageData)
    resizedImageData = transposeDimensions(resizedImageData);
    console.log("Resized image data:", resizedImageData);
    console.log("Shape of the new array:", [resizedImageData.length, resizedImageData[0].length, resizedImageData[0][0].length]);

    // draw slice
    // drawCanvas(canvas, slider.value, niftiHeader, niftiImage);
    drawCanvas2(canvas, parseInt(slider.value), newDimensions, resizedImageData);

    //prediction with onnx model
    var outSegmentationMask = onnxModelPrediction(resizedImageData);
    console.log("output segmentation mask: ", outSegmentationMask)
    

    slider.oninput = function() {
        drawCanvas2(canvas, parseInt(slider.value), newDimensions, resizedImageData);
        // drawCanvas2(canvasOut, parseInt(slider.value), newDimensions, outSegmentationMask);
    };

}

async function onnxModelPrediction(dataArray) {
    
    var model_dims = [64,64,40] // MODIFY : MODEL DIMS
    const myOrtSession = await ort.InferenceSession.create(
        // "nca_model_lungs2.onnx" // MODIFY : MODEL NAME
        "nca_model_liver1.onnx"
        );

    const input0 = new ort.Tensor(
        'float32', dataArray.flat(2),
        [1, model_dims[0], model_dims[1], model_dims[2], 1]
    )
    console.log("input0: ", input0)

    const outputs = await myOrtSession.run({
        'x.1': input0,
        // 'fire_rate' : f2
    })
    
    // const outputTensor = outputs["17"];
    const outputTensor = outputs["3654"];
    var outputTensorData = outputTensor.data;
    console.log(`model output : ${outputTensor}`)
    console.log(`model output tensor: ${outputTensor.data}.`);

    const keys = Object.keys(outputTensor);
    console.log(keys);

    // dims, type, data, size
    console.log(`${outputTensor.dims}, ${outputTensor.type}, ${outputTensor.size}`)
    var outputArray = reshapeFlattenedArray(model_dims, outputTensorData)
    console.log("resized output array dims: ", outputArray.length, outputArray[0].length, outputArray[0][0].length)

    outputArray = applySigmoidAndThreshold(outputArray);


    console.log(outputArray)
    var canvasOut = document.getElementById('myOutCanvas');
    var sliderOut = document.getElementById('myRangeOut');
    var slices = model_dims[2]
    sliderOut.max = slices - 1;
    sliderOut.value = Math.round(slices / 2);
    drawCanvas2(canvasOut, parseInt(sliderOut.value), model_dims, outputArray);

    sliderOut.oninput = function() {
        // drawCanvas2(canvas, parseInt(slider.value), newDimensions, resizedImageData);
        drawCanvas2(canvasOut, parseInt(sliderOut.value), model_dims, outputArray);
    };

    return outputArray


}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function applySigmoidAndThreshold(array3D) {
    const threshold = 0.5;
    const resultArray = [];

    for (let i = 0; i < array3D.length; i++) {
        const slice = [];
        for (let j = 0; j < array3D[i].length; j++) {
            const row = [];
            for (let k = 0; k < array3D[i][j].length; k++) {
                const sigmoidValue = sigmoid(array3D[i][j][k]);
                const binaryValue = sigmoidValue > threshold ? 1 : 0;
                row.push(binaryValue);
            }
            slice.push(row);
        }
        resultArray.push(slice);
    }

    return resultArray;
}

function reshapeFlattenedArray(dimensions, flatData) {
    const [dim1, dim2, dim3] = dimensions;
    const array3D = [];
    let index = 0;
    
    for (let i = 0; i < dim1; i++) {
        const slice = [];
        for (let j = 0; j < dim2; j++) {
            const row = [];
            for (let k = 0; k < dim3; k++) {
                row.push(flatData[index]);
                index++;
            }
            slice.push(row);
        }
        array3D.push(slice);
    }
    return array3D;
}


// Function to calculate mean and standard deviation of an array
function calculateMeanAndStd(array) {
    const mean = array.reduce((acc, val) => acc + val, 0) / array.length;
    const std = Math.sqrt(array.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / array.length);
    return { mean, std };
}

// Z-normalization function
function zNormalize(image) {
    const { mean, std } = calculateMeanAndStd(image.flat());
    return image.map(row => row.map(pixel => (pixel - mean) / std));
}

// Min-max normalization function
function minMaxNormalize(image) {
    const min = Math.min(...image.flat());
    const max = Math.max(...image.flat());
    return image.map(row => row.map(pixel => (pixel - min) / (max - min)));
}

function doNormalization(images) {
    for (let i = 0; i < images.length; i++) {
        images[i] = zNormalize(images[i]);
        images[i] = minMaxNormalize(images[i]);
    }
    return images
}


function getDataArray(niftiHeader, niftiImage) {
    var typedData
    if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
        typedData = new Uint8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
        typedData = new Int16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32) {
        typedData = new Int32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
        typedData = new Float32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) {
        typedData = new Float64Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT8) {
        typedData = new Int8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT16) {
        typedData = new Uint16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT32) {
        typedData = new Uint32Array(niftiImage);
    } else {
        return;
    }
    return typedData;
}

function transposeDimensions(imageData) {
    const [cols, rows, slices] = [imageData[0][0].length, imageData[0].length, imageData.length];
    const transposedImageData = [];

    for (let y = 0; y < rows; y++) {
        const row = [];
        for (let x = 0; x < cols; x++) {
            const col = [];
            for (let z = 0; z < slices; z++) {
                col.push(imageData[z][y][x]);
            }
            row.push(col);
        }
        transposedImageData.push(row);
    }

    return transposedImageData;
}

function resizeImageData(imageData, newDimensions) {
      const [newCols, newRows, newSlices] = newDimensions;
      const [cols, rows, slices] = [imageData[0][0].length, imageData[0].length, imageData.length];

      const resizedImageData = [];
      for (let z = 0; z < newSlices; z++) {
          const zRatio = z / (newSlices - 1) * (slices - 1);
          const slice = [];
          for (let y = 0; y < newRows; y++) {
              const yRatio = y / (newRows - 1) * (rows - 1);
              const row = [];
              for (let x = 0; x < newCols; x++) {
                  const xRatio = x / (newCols - 1) * (cols - 1);
                  const val = trilinearInterpolation(imageData, xRatio, yRatio, zRatio);
                  row.push(val);
              }
              slice.push(row);
          }
          resizedImageData.push(slice);
      }

      return resizedImageData;
}

function trilinearInterpolation(imageData, xRatio, yRatio, zRatio) {
    const [cols, rows, slices] = [imageData[0][0].length, imageData[0].length, imageData.length];
    const x0 = Math.floor(xRatio);
    const y0 = Math.floor(yRatio);
    const z0 = Math.floor(zRatio);
    const x1 = Math.min(x0 + 1, cols - 1);
    const y1 = Math.min(y0 + 1, rows - 1);
    const z1 = Math.min(z0 + 1, slices - 1);

    const xd = xRatio - x0;
    const yd = yRatio - y0;
    const zd = zRatio - z0;

    const c00 = imageData[z0][y0][x0] * (1 - xd) + imageData[z0][y0][x1] * xd;
    const c10 = imageData[z0][y1][x0] * (1 - xd) + imageData[z0][y1][x1] * xd;
    const c01 = imageData[z1][y0][x0] * (1 - xd) + imageData[z1][y0][x1] * xd;
    const c11 = imageData[z1][y1][x0] * (1 - xd) + imageData[z1][y1][x1] * xd;

    const c0 = c00 * (1 - yd) + c10 * yd;
    const c1 = c01 * (1 - yd) + c11 * yd;

    const c = c0 * (1 - zd) + c1 * zd;

    return c;
}

function convertNiftiTo3DArray(niftiHeader, dataArray) {
    const dims = niftiHeader.dims;
    const cols = dims[1];
    const rows = dims[2];
    const slices = dims[3];

    const data = [];
    let dataIndex = 0;

    for (let z = 0; z < slices; z++) {
        const slice = [];
        for (let y = 0; y < rows; y++) {
            const row = [];
            for (let x = 0; x < cols; x++) {
                row.push(dataArray[dataIndex]);
                dataIndex++;
            }
            slice.push(row);
        }
        data.push(slice);
    }

    return data;
}

function drawCanvas2(canvas, slice, newdims, niftiImage) {
    slice = slice % 40
    console.log("slice is : ", slice)
    console.log("Recieved image: ",niftiImage)
    // console.log("Shape of the new array:", [niftiImage.length, niftiImage[0].length, niftiImage[0][0].length]);

    // get nifti dimensions
    var cols = newdims[0];
    var rows = newdims[1];
    // set canvas dimensions to nifti slice dimensions
    canvas.width = cols;
    canvas.height = rows;

    // make canvas image data
    var ctx = canvas.getContext("2d");
    var canvasImageData = ctx.createImageData(canvas.width, canvas.height);

    var typedData = [];
    for (var i = 0; i < niftiImage.length; i++) {
        var row = [];
        for (var j = 0; j < niftiImage[i].length; j++) {
            row.push(niftiImage[i][j][slice]);
        }
        typedData.push(row);
    }


    console.log("typed data: ",typedData)


    // draw pixels
    for (var row = 0; row < rows; row++) {
        var rowOffset = row * cols;
        // var rowOffset = ro;
        for (var col = 0; col < cols; col++) {
            // var offset = sliceOffset + rowOffset + col;
            var value = typedData[row][col];

            canvasImageData.data[(rowOffset + col) * 4] = value*255 & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 1] = value*255 & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 2] = value*255 & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 3] = 0xFF;
        }
    }

    ctx.putImageData(canvasImageData, 0, 0);
}



function drawCanvas(canvas, slice, niftiHeader, niftiImage) {
    // get nifti dimensions
    var cols = niftiHeader.dims[1];
    var rows = niftiHeader.dims[2];

    // set canvas dimensions to nifti slice dimensions
    canvas.width = cols;
    canvas.height = rows;

    // make canvas image data
    var ctx = canvas.getContext("2d");
    var canvasImageData = ctx.createImageData(canvas.width, canvas.height);

    // convert raw data to typed array based on nifti datatype
    var typedData;

    if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT8) {
        typedData = new Uint8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT16) {
        typedData = new Int16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT32) {
        typedData = new Int32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) {
        typedData = new Float32Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_FLOAT64) {
        typedData = new Float64Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_INT8) {
        typedData = new Int8Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT16) {
        typedData = new Uint16Array(niftiImage);
    } else if (niftiHeader.datatypeCode === nifti.NIFTI1.TYPE_UINT32) {
        typedData = new Uint32Array(niftiImage);
    } else {
        return;
    }

    // offset to specified slice
    var sliceSize = cols * rows;
    var sliceOffset = sliceSize * slice;

    // draw pixels
    for (var row = 0; row < rows; row++) {
        var rowOffset = row * cols;

        for (var col = 0; col < cols; col++) {
            var offset = sliceOffset + rowOffset + col;
            var value = typedData[offset];

            canvasImageData.data[(rowOffset + col) * 4] = value & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 1] = value & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 2] = value & 0xFF;
            canvasImageData.data[(rowOffset + col) * 4 + 3] = 0xFF;
        }
    }

    ctx.putImageData(canvasImageData, 0, 0);
}

function makeSlice(file, start, length) {
    var fileType = (typeof File);

    if (fileType === 'undefined') {
        return function () {};
    }

    if (File.prototype.slice) {
        return file.slice(start, start + length);
    }

    if (File.prototype.mozSlice) {
        return file.mozSlice(start, length);
    }

    if (File.prototype.webkitSlice) {
        return file.webkitSlice(start, length);
    }

    return null;
}

function readFile(file) {
    var blob = makeSlice(file, 0, file.size);

    var reader = new FileReader();

    reader.onloadend = function (evt) {
        if (evt.target.readyState === FileReader.DONE) {
            readNIFTI(file.name, evt.target.result);
        }
    };

    reader.readAsArrayBuffer(blob);
}

function handleFileSelect(evt) {
    var files = evt.target.files;
    readFile(files[0]);
}

document.getElementById('file').addEventListener('change', handleFileSelect, false);