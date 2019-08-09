from string import Template
import sys, os
import re 
import tensorflow as tf
import os
pypath = os.path.dirname(os.path.realpath(__file__))

supported_types = ["float", "double", "int32", "bool", "string"]
used_names = set([])

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def typeerror(tp):
    if tp not in supported_types:
        print("ERROR: type {} not supported\nSupported types: {}".format(tp, supported_types))
        exit(0)
    return tp

def parsevariable(var):
    varname, remaining = var.split("(")
    if not varname.islower():
        print("ERROR: variable names must all be lower case.")
        exit(0)
    if varname in used_names:
        print("ERROR: variable name {} has already been used ==> {}".format(varname, used_names))
        exit(0)
    else:
        used_names.add(varname)
        used_names.add("grad_"+varname)
    dims = remaining.strip(")").split(",")
    if len(dims)==1 and len(dims[0])==0:
        return varname, 0, None
    for i in range(len(dims)):
        if dims[i]=="?":
            dims[i]=-1
        else:
            dims[i]=int(dims[i])
    return varname, len(dims), dims
############################ Common ############################
AttributesReg_T = Template("""${nn}.Attr("${name}: ${tp}")""")
def AttributesReg():
    s = ""
    for k,item in enumerate(attributes):
        if item[0]=="double":
            print("ERROR: Attributes only accept float")
            exit(0)
        s += AttributesReg_T.substitute({"name":item[1], "tp": item[0],
                    "nn":"" if k==0 else "\n  "})
    return s

AttributesParse_T1 = Template("""
OP_REQUIRES_OK(context, c->GetAttr("${name}", &${name}_));""")
AttributesParse_T2 = Template("""
OP_REQUIRES_OK(context, context->GetAttr("${name}", &${name}_));""")
def AttributesParse1():
    s = ""
    for item in attributes:
        s += AttributesParse_T1.substitute({"name":item[1]})
    return s
def AttributesParse2():
    s = ""
    for item in attributes:
        s += AttributesParse_T2.substitute({"name":item[1]})
    return s

AttributesDef_T = Template("""    ${tp} ${name}_;\n""")
def AttributesDef():
    s = ""
    for item in attributes:
        s += AttributesDef_T.substitute({"name":item[1], "tp": item[0]})
    return s


############################ Forward ############################
def ForwardInputOutput():
    s = ""
    for i in range(len(inputs)):
        s += "{}.Input(\"{} : {}\")\n".format("" if i==0 else "  ", inputs[i][1], inputs[i][0])
    for i in range(len(outputs)):
        s += "  .Output(\"{} : {}\"){}".format(outputs[i][1], outputs[i][0], "" if len(outputs)-1==i else "\n")
    return s

SetShapeFn_T1 = Template("""
        shape_inference::ShapeHandle ${name}_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(${id}), ${rank}, &${name}_shape));""")
SetShapeFn_T2 = Template("""
        c->set_output(${id}, c->${SVM}(${dims}));""")
    
def SetShapeFn():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]
        s += SetShapeFn_T1.substitute({"name":item[1], "id":i, "rank":item[2]})
    s += "\n"
    for i in range(len(outputs)):
        item = outputs[i]
        if item[2]==0:
            s+="\n        c->set_output({}, c->Scalar());".format(i)
        elif item[2]==1:
            s+=SetShapeFn_T2.substitute({"name":item[1], "id":i, "dims":",".join([str(x) for x in item[3]]), "SVM":"Vector"})
        elif item[2]==2:
            s+=SetShapeFn_T2.substitute({"name":item[1], "id":i, "dims":",".join([str(x) for x in item[3]]), "SVM":"Matrix"})
        elif item[2]==3:
            s+=SetShapeFn_T2.substitute({"name":item[1], "id":i, "dims":",".join([str(x) for x in item[3]]), "SVM":"Tensor"})
    return s

ForwardTensor_T = Template("""
    const Tensor& ${name} = context->input(${id});""")
def ForwardTensor():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]
        s += ForwardTensor_T.substitute({"name": item[1], "id":i})
    return s

ForwardTensorShape_T = Template("""
    const TensorShape& ${name}_shape = ${name}.shape();""")
def ForwardTensorShape():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]
        if item[0]=="string":
            continue
        s += ForwardTensorShape_T.substitute({"name": item[1]})
    return s

ForwardCheckShape_T = Template("""
    DCHECK_EQ(${name}_shape.dims(), ${dims});""")
def ForwardCheckShape():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]
        if item[0]=="string":
            continue
        s += ForwardCheckShape_T.substitute({"name": item[1], "dims": item[2]})
    return s


ForwardOutputShape_T = Template("""
    TensorShape ${name}_shape(${dims});""")
def ForwardOutputShape():
    s = ""
    for i in range(len(outputs)):
        item = outputs[i]
        if item[0]=="string":
            continue
        if item[3]==None:
            dims = "{}"
        else:
            dims = "{"+",".join([str(x) for x in item[3]])+"}"
        s += ForwardOutputShape_T.substitute({"name": item[1], "dims": dims})
    return s

ForwardOutput_T = Template("""
    Tensor* ${name} = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(${id}, ${name}_shape, &${name}));""")
def ForwardOutput():
    s = ""
    for i in range(len(outputs)):
        item = outputs[i]
        if item[0]=="string":
            continue
        s += ForwardOutput_T.substitute({"name": item[1], "id": i})
    return s

ForwardGetData_T1 = Template("""
    auto ${name}_tensor = ${name}.flat<${tp}>().data();""")
ForwardGetData_T12 = Template("""
    string ${name}_tensor = string(*${name}.flat<${tp}>().data());""")
ForwardGetData_T2 = Template("""
    auto ${name}_tensor = ${name}->flat<${tp}>().data();""")
ForwardGetData_T3 = Template("""
    string ${name}_tensor = string(*${name}->flat<${tp}>().data());""")
def ForwardGetData():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]
        if item[0]=="string":
            s += ForwardGetData_T12.substitute({"name": item[1], "tp": item[0]})
        else:
            s += ForwardGetData_T1.substitute({"name": item[1], "tp": item[0]})
    for i in range(len(outputs)):
        item = outputs[i]
        if item[0]=="string":
            s += ForwardGetData_T3.substitute({"name": item[1], "tp": item[0]})
        else:
            s += ForwardGetData_T2.substitute({"name": item[1], "tp": item[0]})
    return s 

############################ Backward ############################
def BackwardInputOutput():
    s = ""
    for i in range(len(outputs)):
        s += ".Input(\"grad_{} : {}\")\n".format(outputs[i][1], outputs[i][0])
    for i in range(len(outputs)):
        s += "{}.Input(\"{} : {}\")\n".format("  ", outputs[i][1], outputs[i][0])
    for i in range(len(inputs)):
        s += "  .Input(\"{} : {}\")\n".format(inputs[i][1], inputs[i][0])
    for i in range(len(inputs)):
        s += "  .Output(\"grad_{} : {}\"){}".format(inputs[i][1], inputs[i][0], ";" if i==len(inputs)-1 else "\n")
    return s


BackwardTensor_T1 = Template("""
    const Tensor& ${name} = context->input(${id});""")
BackwardTensor_T2 = Template("""
    const Tensor& grad_${name} = context->input(${id});""")
def BackwardTensor():
    s = ""
    for i in range(len(outputs)):
        s += BackwardTensor_T2.substitute({"name": outputs[i][1], "id":i})
    for i in range(len(outputs)):
        item = outputs[i]
        s += BackwardTensor_T1.substitute({"name": item[1], "id":i+len(outputs)})
    for i in range(len(inputs)):
        item = inputs[i]
        s += BackwardTensor_T1.substitute({"name": item[1], "id":i+len(outputs)+len(outputs)})
    return s

BackwardTensorShape_T1 = Template("""
    const TensorShape& ${name}_shape = ${name}.shape();""")
BackwardTensorShape_T2 = Template("""
    const TensorShape& grad_${name}_shape = grad_${name}.shape();""")
def BackwardTensorShape():
    s = ""
    for i in range(len(outputs)):
        if outputs[i][0]=="string":
            continue
        s += BackwardTensorShape_T2.substitute({"name": outputs[i][1]})
    for i in range(len(outputs)):
        if outputs[i][0]=="string":
            continue
        item = outputs[i]
        s += BackwardTensorShape_T1.substitute({"name": item[1]})
    for i in range(len(inputs)):
        if inputs[i][0]=="string":
            continue
        item = inputs[i]
        s += BackwardTensorShape_T1.substitute({"name": item[1]})
    return s

BackwardCheckShape_T1 = Template("""
    DCHECK_EQ(${name}_shape.dims(), ${dims});""")
BackwardCheckShape_T2 = Template("""
    DCHECK_EQ(grad_${name}_shape.dims(), ${dims});""")
def BackwardCheckShape():
    s = ""
    for i in range(len(outputs)):
        if outputs[i][0]=="string":
            continue
        s += BackwardCheckShape_T2.substitute({"name": outputs[i][1], "dims": outputs[i][2]})
    for i in range(len(outputs)):
        if outputs[i][0]=="string":
            continue
        item = outputs[i]
        s += BackwardCheckShape_T1.substitute({"name": item[1], "dims": item[2]})
    for i in range(len(inputs)):
        if inputs[i][0]=="string":
            continue
        item = inputs[i]
        s += BackwardCheckShape_T1.substitute({"name": item[1], "dims": item[2]})
    return s


BackwardOutputShape_T = Template("""
    TensorShape grad_${name}_shape(${name}_shape);""")
def BackwardOutputShape():
    s = ""
    for i in range(len(inputs)):
        if inputs[i][0]=="string":
            continue
        item = inputs[i]
        if item[3]==None:
            dims = "{}"
        else:
            dims = "{" + ",".join([str(x) for x in item[3]]) + "}"
        s += BackwardOutputShape_T.substitute({"name": item[1], "dims": dims})
    return s

BackwardOutput_T = Template("""
    Tensor* grad_${name} = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(${id}, grad_${name}_shape, &grad_${name}));""")
def BackwardOutput():
    s = ""
    for i in range(len(inputs)):
        if inputs[i][0]=="string":
            continue
        item = inputs[i]
        s += BackwardOutput_T.substitute({"name": item[1], "id":i})
    return s

BackwardGetData_Input_T1 = Template("""
    auto ${name}_tensor = ${name}.flat<${tp}>().data();""")
BackwardGetData_Input_T2 = Template("""
    auto grad_${name}_tensor = grad_${name}.flat<${tp}>().data();""")
BackwardGetData_Output = Template("""
    auto grad_${name}_tensor = grad_${name}->flat<${tp}>().data();""")

BackwardGetData_Input_T1_ = Template("""
    auto ${name}_tensor = string(*${name}.flat<${tp}>().data());""")
BackwardGetData_Input_T2_ = Template("""""")
BackwardGetData_Output_ = Template("""""")
def BackwardGetData():
    s = ""
    for i in range(len(inputs)):
        item = inputs[i]

        if item[0]=="string":
            continue
        else:
            s += BackwardGetData_Input_T1.substitute({"name": item[1], "tp": item[0]})
    for i in range(len(outputs)):
        if item[0]=="string":
            continue
        else:
            s += BackwardGetData_Input_T2.substitute({"name": outputs[i][1], "tp": outputs[i][0]})
    for i in range(len(outputs)):
        item = outputs[i]
        if item[0]=="string":
            s += BackwardGetData_Input_T1_.substitute({"name": item[1], "tp": item[0]})
        else:
            s += BackwardGetData_Input_T1.substitute({"name": item[1], "tp": item[0]})
    for i in range(len(inputs)):
        item = inputs[i]
        if item[0]=="string":
            s += BackwardGetData_Output_.substitute({"name": item[1], "tp": item[0]})
        else:
            s += BackwardGetData_Output.substitute({"name": item[1], "tp": item[0]})
    return s 

def GetAttr():
    s = "        attr_dict={{}}{}".format("" if len(attributes)==0 else "\n")
    for item in attributes:
        s += """        attr_dict[\"{}\"]=op.get_attr(\"{}\")\n""".format(item[1], item[1])
    return s

def ARGS():
    s = ",".join([item[1] for item in inputs])
    if len(attributes)>0:
        s += ","+",".join(["{}={}".format(item[1],item[1]) for item in attributes])
    return s

def OUTPUT():
    s = ",".join([item[1] for item in outputs])
    return s

filename = sys.argv[1]
dirname = sys.argv[2]
iftorch = sys.argv[3]
if filename not in os.listdir("."):
    print("ERROR: file {} does not exist".format(filename))
    exit(0)
inputs = []
outputs = []
attributes = []

with open(filename, "r") as fp:
    op = fp.readline().strip()
    for line in fp:
        items = line.strip().split()
        if len(items)!=2 and len(items)!=4:
            continue
        
        if len(items)==2:
            tp = typeerror(items[0])
            varname, rank, dims = parsevariable(items[1])
            inputs.append([tp, varname, rank, dims])
        else:
            tp = typeerror(items[0])
            varname, rank, dims = parsevariable(items[1])
            assert(items[2]=="->")
            if items[3]=="output":
                outputs.append([tp, varname, rank, dims])
            elif items[3]=="attribute":
                attributes.append([tp, varname])
            else:
                print("ERROR: unidentifiable featuer --> {}".format(items[3]))

# print(inputs)
# print(outputs)
# print(ForwardInputOutput())
# print(SetShapeFn())

d = {"OperatorName": op,
    "SetShapeFn": SetShapeFn(),
    "ForwardInputOutput": ForwardInputOutput(),
    "BackwardInputOutput": BackwardInputOutput(),
    "ForwarInputNum": len(inputs),
    "ForwardTensor": ForwardTensor(),
    "ForwardTensorShape": ForwardTensorShape(),
    "ForwardCheckShape": ForwardCheckShape(),
    "ForwardOutputShape": ForwardOutputShape(),
    "ForwardOutput": ForwardOutput(),
    "ForwardGetData": ForwardGetData(),
    "BackwardInputOutput": BackwardInputOutput(),
    "BackwardTensor": BackwardTensor(),
    "BackwardTensorShape": BackwardTensorShape(),
    "BackwardCheckShape": BackwardCheckShape(),
    "BackwardOutputShape": BackwardOutputShape(),
    "BackwardOutput": BackwardOutput(),
    "BackwardGetData": BackwardGetData(),
    "AttributesReg": AttributesReg(),
    "AttributesDef": AttributesDef(),
    "AttributesParse2": AttributesParse2(),
    "GetAttr": GetAttr()}

with open("{}/custom_op.template".format(dirname),"r") as fp:
    cnt = fp.read()
    s = Template(cnt)
    cpp = s.substitute(d)

with open("{}.cpp".format(op), "w") as fp:
    fp.write(cpp)

d = {"include": tf.sysconfig.get_compile_flags()[0][2:],
    "link": tf.sysconfig.get_link_flags()[0][2:],
    "ABI": tf.sysconfig.get_compile_flags()[1][-1],
    "OperatorName": op,
    "IfTorch": iftorch,
    "HEADERS": os.path.join(pypath, "../headers")}
with open("{}/CMakeLists.template".format(dirname),"r") as fp:
    cnt = fp.read()
    s = Template(cnt)
    cmake = s.substitute(d)
with open("CMakeLists.txt", "w") as fp:
    fp.write(cmake)

import platform
d = {"operator_name": convert(op),
    "dylibso": "dylib" if platform.system()=="Darwin" else "so",
    "OperatorName": op,
    "GetAttr": GetAttr(),
    "ARGS": ARGS(),
    "OUTPUT": OUTPUT(),
    "FIRST_OUTPUT": outputs[0][1]}
with open("{}/gradtest.template".format(dirname),"r") as fp:
    cnt = fp.read()
    s = Template(cnt)
    jl = s.substitute(d)
with open("gradtest.jl", "w") as fp:
    fp.write(jl)
