**Set output shape**
```
c->set_output(0, c->Vector(n));
c->set_output(0, c->Matrix(m, n));
c->set_output(0, c->Scalar());
```

**Names**
`.Input` and `.Ouput` : names must be in lower case, no `_`, only letters.

**TensorFlow Input/Output to TensorFlow Tensors**
```
grad.vec<double>();
grad.scalar<double>();
grad.matrix<double>();
grad.flat<double>();
```
Obtain flat arrays
```
grad.flat<double>().data()
```

**Scalars**
Allocate scalars using TensorShape()

**Allocate Shapes**
Although you can use -1 for shape reference, you must allocate exact shapes in `Compute`