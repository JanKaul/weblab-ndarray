import {Ndarray} from "weblab-ndarray";

// var array1 = new Float64Array(1048576);
// var array2 = new Float64Array(1048576);
//
// for (let i = 0; i<1048576; i++){
//   array1[i] = Math.random();
//   array2[i] = Math.random();
// }
//
// var wasm1 = new Ndarray(array1);
var wasm2 = new Ndarray([[11,12,13],[21,22,23],[31,32,33]]);

console.log(wasm2.slices([[0,2],[0,2]]).get([1,1]));
