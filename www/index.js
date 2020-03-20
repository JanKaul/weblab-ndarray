import {Ndarray} from "weblab-ndarray";

var array1 = new Float64Array(1048576);
var array2 = new Float64Array(1048576);

for (let i = 0; i<1048576; i++){
  array1[i] = Math.random();
  array2[i] = Math.random();
}

let wasm1 = new Ndarray(array1);
