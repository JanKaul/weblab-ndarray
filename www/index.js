import {TestArray} from "weblab-ndarray";

var array1 = new Float64Array(1048576);
var array2 = new Float64Array(1048576);

for (let i = 0; i<1048576; i++){
  array1[i] = Math.random();
  array2[i] = Math.random();
}

let wasm1 = new TestArray(array1);
let wasm2 = new TestArray(array2);

var array3 = new Float64Array(1048576);

let t1 = performance.now();

for (let i = 0; i<1048576; i++){
  array3[i] = array1[i] * array2[i];
}

let t2 = performance.now();

let wasm3 = wasm1.mul(wasm2);

let t3 = performance.now();

let t4 = t2 - t1;

let t5 = t3 - t2;

console.log(t2 - t1);
console.log(t3 - t2);
