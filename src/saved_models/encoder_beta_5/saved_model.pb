ó
É­
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
,
Exp
x"T
y"T"
Ttype:

2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02unknown8î


multi_graph_cnn/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_namemulti_graph_cnn/kernel

*multi_graph_cnn/kernel/Read/ReadVariableOpReadVariableOpmulti_graph_cnn/kernel*
_output_shapes

:d*
dtype0

multi_graph_cnn/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_namemulti_graph_cnn/bias
y
(multi_graph_cnn/bias/Read/ReadVariableOpReadVariableOpmulti_graph_cnn/bias*
_output_shapes
:d*
dtype0

multi_graph_cnn_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Èd*)
shared_namemulti_graph_cnn_1/kernel

,multi_graph_cnn_1/kernel/Read/ReadVariableOpReadVariableOpmulti_graph_cnn_1/kernel*
_output_shapes
:	Èd*
dtype0

multi_graph_cnn_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_namemulti_graph_cnn_1/bias
}
*multi_graph_cnn_1/bias/Read/ReadVariableOpReadVariableOpmulti_graph_cnn_1/bias*
_output_shapes
:d*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0

NoOpNoOp
:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ï9
valueÅ9BÂ9 B»9
÷
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¥
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses* 
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
¥
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses* 

3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
¦

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
¦

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*
¦

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
¦

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses*

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
Z
0
1
$2
%3
94
:5
A6
B7
I8
J9
Q10
R11*
Z
0
1
$2
%3
94
:5
A6
B7
I8
J9
Q10
R11*
* 
°
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

dserving_default* 
f`
VARIABLE_VALUEmulti_graph_cnn/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEmulti_graph_cnn/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 
* 
* 
* 
hb
VARIABLE_VALUEmulti_graph_cnn_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEmulti_graph_cnn_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 
* 
* 
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

 serving_default_adjacency_matrixPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

serving_default_node_attributesPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
ô
StatefulPartitionedCallStatefulPartitionedCall serving_default_adjacency_matrixserving_default_node_attributesmulti_graph_cnn/kernelmulti_graph_cnn/biasmulti_graph_cnn_1/kernelmulti_graph_cnn_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_3709
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*multi_graph_cnn/kernel/Read/ReadVariableOp(multi_graph_cnn/bias/Read/ReadVariableOp,multi_graph_cnn_1/kernel/Read/ReadVariableOp*multi_graph_cnn_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_4063
ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemulti_graph_cnn/kernelmulti_graph_cnn/biasmulti_graph_cnn_1/kernelmulti_graph_cnn_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_4109¢º	
¸	

C__inference_z_log_var_layer_call_and_return_conditional_losses_3945

inputs8
&matmul_readvariableop_z_log_var_kernel:3
%biasadd_readvariableop_z_log_var_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_z_log_var_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_z_log_var_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
­
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703

inputs
inputs_1B
/shape_2_readvariableop_multi_graph_cnn_1_kernel:	Èd;
-biasadd_readvariableop_multi_graph_cnn_1_bias:d
identity¢BiasAdd/ReadVariableOp¢transpose/ReadVariableOp_
MatMulBatchMatMulV2inputs_1inputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:1concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈF
Shape_1Shapeconcat:output:0*
T0*
_output_shapes
:S
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num
Shape_2/ReadVariableOpReadVariableOp/shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"È   d   S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   n
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
transpose/ReadVariableOpReadVariableOp/shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes
:	Èd`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"È   ÿÿÿÿg
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	Èdj
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_multi_graph_cnn_1_bias*
_output_shapes
:d*
dtype0|
BiasAddBiasAddReshape_2:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
NoOpNoOp^BiasAdd/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«z
ç

A__inference_encoder_layer_call_and_return_conditional_losses_3553
inputs_0
inputs_1O
=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel:dI
;multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias:dT
Amulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel:	ÈdM
?multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias:d:
(dense_matmul_readvariableop_dense_kernel:d5
'dense_biasadd_readvariableop_dense_bias:>
,dense_1_matmul_readvariableop_dense_1_kernel:9
+dense_1_biasadd_readvariableop_dense_1_bias:<
*z_mean_matmul_readvariableop_z_mean_kernel:7
)z_mean_biasadd_readvariableop_z_mean_bias:B
0z_log_var_matmul_readvariableop_z_log_var_kernel:=
/z_log_var_biasadd_readvariableop_z_log_var_bias:
identity

identity_1

identity_2¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢&multi_graph_cnn/BiasAdd/ReadVariableOp¢(multi_graph_cnn/transpose/ReadVariableOp¢(multi_graph_cnn_1/BiasAdd/ReadVariableOp¢*multi_graph_cnn_1/transpose/ReadVariableOp¢ z_log_var/BiasAdd/ReadVariableOp¢z_log_var/MatMul/ReadVariableOp¢z_mean/BiasAdd/ReadVariableOp¢z_mean/MatMul/ReadVariableOpq
multi_graph_cnn/MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
multi_graph_cnn/ShapeShapemulti_graph_cnn/MatMul:output:0*
T0*
_output_shapes
:a
multi_graph_cnn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ç
multi_graph_cnn/splitSplit(multi_graph_cnn/split/split_dim:output:0multi_graph_cnn/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split]
multi_graph_cnn/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
multi_graph_cnn/concatConcatV2multi_graph_cnn/split:output:0multi_graph_cnn/split:output:1$multi_graph_cnn/concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
multi_graph_cnn/Shape_1Shapemulti_graph_cnn/concat:output:0*
T0*
_output_shapes
:s
multi_graph_cnn/unstackUnpack multi_graph_cnn/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num¤
&multi_graph_cnn/Shape_2/ReadVariableOpReadVariableOp=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0h
multi_graph_cnn/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   d   s
multi_graph_cnn/unstack_1Unpack multi_graph_cnn/Shape_2:output:0*
T0*
_output_shapes
: : *	
numn
multi_graph_cnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
multi_graph_cnn/ReshapeReshapemulti_graph_cnn/concat:output:0&multi_graph_cnn/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
(multi_graph_cnn/transpose/ReadVariableOpReadVariableOp=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0o
multi_graph_cnn/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ª
multi_graph_cnn/transpose	Transpose0multi_graph_cnn/transpose/ReadVariableOp:value:0'multi_graph_cnn/transpose/perm:output:0*
T0*
_output_shapes

:dp
multi_graph_cnn/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
multi_graph_cnn/Reshape_1Reshapemulti_graph_cnn/transpose:y:0(multi_graph_cnn/Reshape_1/shape:output:0*
T0*
_output_shapes

:d
multi_graph_cnn/MatMul_1MatMul multi_graph_cnn/Reshape:output:0"multi_graph_cnn/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!multi_graph_cnn/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!multi_graph_cnn/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :dÏ
multi_graph_cnn/Reshape_2/shapePack multi_graph_cnn/unstack:output:0*multi_graph_cnn/Reshape_2/shape/1:output:0*multi_graph_cnn/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:¨
multi_graph_cnn/Reshape_2Reshape"multi_graph_cnn/MatMul_1:product:0(multi_graph_cnn/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&multi_graph_cnn/BiasAdd/ReadVariableOpReadVariableOp;multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias*
_output_shapes
:d*
dtype0¬
multi_graph_cnn/BiasAddBiasAdd"multi_graph_cnn/Reshape_2:output:0.multi_graph_cnn/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
multi_graph_cnn/EluElu multi_graph_cnn/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
dropout/IdentityIdentity!multi_graph_cnn/Elu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
multi_graph_cnn_1/MatMulBatchMatMulV2inputs_1dropout/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
multi_graph_cnn_1/ShapeShape!multi_graph_cnn_1/MatMul:output:0*
T0*
_output_shapes
:c
!multi_graph_cnn_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Í
multi_graph_cnn_1/splitSplit*multi_graph_cnn_1/split/split_dim:output:0!multi_graph_cnn_1/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split_
multi_graph_cnn_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ð
multi_graph_cnn_1/concatConcatV2 multi_graph_cnn_1/split:output:0 multi_graph_cnn_1/split:output:1&multi_graph_cnn_1/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
multi_graph_cnn_1/Shape_1Shape!multi_graph_cnn_1/concat:output:0*
T0*
_output_shapes
:w
multi_graph_cnn_1/unstackUnpack"multi_graph_cnn_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num«
(multi_graph_cnn_1/Shape_2/ReadVariableOpReadVariableOpAmulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0j
multi_graph_cnn_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"È   d   w
multi_graph_cnn_1/unstack_1Unpack"multi_graph_cnn_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
nump
multi_graph_cnn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   ¤
multi_graph_cnn_1/ReshapeReshape!multi_graph_cnn_1/concat:output:0(multi_graph_cnn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ­
*multi_graph_cnn_1/transpose/ReadVariableOpReadVariableOpAmulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0q
 multi_graph_cnn_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ±
multi_graph_cnn_1/transpose	Transpose2multi_graph_cnn_1/transpose/ReadVariableOp:value:0)multi_graph_cnn_1/transpose/perm:output:0*
T0*
_output_shapes
:	Èdr
!multi_graph_cnn_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"È   ÿÿÿÿ
multi_graph_cnn_1/Reshape_1Reshapemulti_graph_cnn_1/transpose:y:0*multi_graph_cnn_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Èd 
multi_graph_cnn_1/MatMul_1MatMul"multi_graph_cnn_1/Reshape:output:0$multi_graph_cnn_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
#multi_graph_cnn_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#multi_graph_cnn_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d×
!multi_graph_cnn_1/Reshape_2/shapePack"multi_graph_cnn_1/unstack:output:0,multi_graph_cnn_1/Reshape_2/shape/1:output:0,multi_graph_cnn_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:®
multi_graph_cnn_1/Reshape_2Reshape$multi_graph_cnn_1/MatMul_1:product:0*multi_graph_cnn_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
(multi_graph_cnn_1/BiasAdd/ReadVariableOpReadVariableOp?multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias*
_output_shapes
:d*
dtype0²
multi_graph_cnn_1/BiasAddBiasAdd$multi_graph_cnn_1/Reshape_2:output:00multi_graph_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
multi_graph_cnn_1/EluElu"multi_graph_cnn_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
dropout_1/IdentityIdentity#multi_graph_cnn_1/Elu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda/MeanMeandropout_1/Identity:output:0&lambda/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:d*
dtype0
dense/MatMulMatMullambda/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_mean/MatMul/ReadVariableOpReadVariableOp*z_mean_matmul_readvariableop_z_mean_kernel*
_output_shapes

:*
dtype0
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_mean/BiasAdd/ReadVariableOpReadVariableOp)z_mean_biasadd_readvariableop_z_mean_bias*
_output_shapes
:*
dtype0
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_log_var/MatMul/ReadVariableOpReadVariableOp0z_log_var_matmul_readvariableop_z_log_var_kernel*
_output_shapes

:*
dtype0
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp/z_log_var_biasadd_readvariableop_z_log_var_bias*
_output_shapes
:*
dtype0
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: a
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:Y
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    [
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?º
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ùÙú
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^multi_graph_cnn/BiasAdd/ReadVariableOp)^multi_graph_cnn/transpose/ReadVariableOp)^multi_graph_cnn_1/BiasAdd/ReadVariableOp+^multi_graph_cnn_1/transpose/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&multi_graph_cnn/BiasAdd/ReadVariableOp&multi_graph_cnn/BiasAdd/ReadVariableOp2T
(multi_graph_cnn/transpose/ReadVariableOp(multi_graph_cnn/transpose/ReadVariableOp2T
(multi_graph_cnn_1/BiasAdd/ReadVariableOp(multi_graph_cnn_1/BiasAdd/ReadVariableOp2X
*multi_graph_cnn_1/transpose/ReadVariableOp*multi_graph_cnn_1/transpose/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

D
(__inference_dropout_1_layer_call_fn_3831

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2712d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3028

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö
Å
.__inference_multi_graph_cnn_layer_call_fn_3717
inputs_0
inputs_1(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1multi_graph_cnn_kernelmulti_graph_cnn_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¾
¨
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_3754
inputs_0
inputs_1?
-shape_2_readvariableop_multi_graph_cnn_kernel:d9
+biasadd_readvariableop_multi_graph_cnn_bias:d
identity¢BiasAdd/ReadVariableOp¢transpose/ReadVariableOpa
MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:1concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
Shape_1Shapeconcat:output:0*
T0*
_output_shapes
:S
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num
Shape_2/ReadVariableOpReadVariableOp-shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   d   S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   m
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
transpose/ReadVariableOpReadVariableOp-shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:d`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dj
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_multi_graph_cnn_bias*
_output_shapes
:d*
dtype0|
BiasAddBiasAddReshape_2:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
NoOpNoOp^BiasAdd/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ò
_
&__inference_dropout_layer_call_fn_3764

inputs
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_3114s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¢

ö
?__inference_dense_layer_call_and_return_conditional_losses_2733

inputs4
"matmul_readvariableop_dense_kernel:d/
!biasadd_readvariableop_dense_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ó	
ù
@__inference_z_mean_layer_call_and_return_conditional_losses_2762

inputs5
#matmul_readvariableop_z_mean_kernel:0
"biasadd_readvariableop_z_mean_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_z_mean_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_z_mean_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
;__inference_z_layer_call_and_return_conditional_losses_2802

inputs
inputs_1
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2þÌ>
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
i
 __inference_z_layer_call_fn_3957
inputs_0
inputs_1
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
º
_
A__inference_dropout_layer_call_and_return_conditional_losses_3769

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¨6
æ
A__inference_encoder_layer_call_and_return_conditional_losses_3234

inputs
inputs_18
&multi_graph_cnn_multi_graph_cnn_kernel:d2
$multi_graph_cnn_multi_graph_cnn_bias:d=
*multi_graph_cnn_1_multi_graph_cnn_1_kernel:	Èd6
(multi_graph_cnn_1_multi_graph_cnn_1_bias:d$
dense_dense_kernel:d
dense_dense_bias:(
dense_1_dense_1_kernel:"
dense_1_dense_1_bias:&
z_mean_z_mean_kernel: 
z_mean_z_mean_bias:,
z_log_var_z_log_var_kernel:&
z_log_var_z_log_var_bias:
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢'multi_graph_cnn/StatefulPartitionedCall¢)multi_graph_cnn_1/StatefulPartitionedCall¢z/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCallº
'multi_graph_cnn/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1&multi_graph_cnn_multi_graph_cnn_kernel$multi_graph_cnn_multi_graph_cnn_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655õ
dropout/StatefulPartitionedCallStatefulPartitionedCall0multi_graph_cnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_3114è
)multi_graph_cnn_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0inputs_1*multi_graph_cnn_1_multi_graph_cnn_1_kernel(multi_graph_cnn_1_multi_graph_cnn_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2multi_graph_cnn_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3028Ù
lambda/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2993
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2733
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2748
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_z_mean_kernelz_mean_z_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_2762©
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_z_log_var_kernelz_log_var_z_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_2776­
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2858v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall(^multi_graph_cnn/StatefulPartitionedCall*^multi_graph_cnn_1/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2R
'multi_graph_cnn/StatefulPartitionedCall'multi_graph_cnn/StatefulPartitionedCall2V
)multi_graph_cnn_1/StatefulPartitionedCall)multi_graph_cnn_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_3841

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
µ
£
(__inference_z_log_var_layer_call_fn_3935

inputs"
z_log_var_kernel:
z_log_var_bias:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputsz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_2776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

B
&__inference_dropout_layer_call_fn_3759

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2664d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¬

ü
A__inference_dense_1_layer_call_and_return_conditional_losses_2748

inputs6
$matmul_readvariableop_dense_1_kernel:1
#biasadd_readvariableop_dense_1_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
¦
&__inference_encoder_layer_call_fn_3435
inputs_0
inputs_1(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
dense_kernel:d

dense_bias: 
dense_1_kernel:
dense_1_bias:
z_mean_kernel:
z_mean_bias:"
z_log_var_kernel:
z_log_var_bias:
identity

identity_1

identity_2¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1multi_graph_cnn_kernelmulti_graph_cnn_biasmulti_graph_cnn_1_kernelmulti_graph_cnn_1_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasz_mean_kernelz_mean_biasz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_3234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
»6
÷
A__inference_encoder_layer_call_and_return_conditional_losses_3391
node_attributes
adjacency_matrix8
&multi_graph_cnn_multi_graph_cnn_kernel:d2
$multi_graph_cnn_multi_graph_cnn_bias:d=
*multi_graph_cnn_1_multi_graph_cnn_1_kernel:	Èd6
(multi_graph_cnn_1_multi_graph_cnn_1_bias:d$
dense_dense_kernel:d
dense_dense_bias:(
dense_1_dense_1_kernel:"
dense_1_dense_1_bias:&
z_mean_z_mean_kernel: 
z_mean_z_mean_bias:,
z_log_var_z_log_var_kernel:&
z_log_var_z_log_var_bias:
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢'multi_graph_cnn/StatefulPartitionedCall¢)multi_graph_cnn_1/StatefulPartitionedCall¢z/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCallË
'multi_graph_cnn/StatefulPartitionedCallStatefulPartitionedCallnode_attributesadjacency_matrix&multi_graph_cnn_multi_graph_cnn_kernel$multi_graph_cnn_multi_graph_cnn_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655õ
dropout/StatefulPartitionedCallStatefulPartitionedCall0multi_graph_cnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_3114ð
)multi_graph_cnn_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0adjacency_matrix*multi_graph_cnn_1_multi_graph_cnn_1_kernel(multi_graph_cnn_1_multi_graph_cnn_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2multi_graph_cnn_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3028Ù
lambda/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2993
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2733
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2748
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_z_mean_kernelz_mean_z_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_2762©
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_z_log_var_kernelz_log_var_z_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_2776­
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2858v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall(^multi_graph_cnn/StatefulPartitionedCall*^multi_graph_cnn_1/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2R
'multi_graph_cnn/StatefulPartitionedCall'multi_graph_cnn/StatefulPartitionedCall2V
)multi_graph_cnn_1/StatefulPartitionedCall)multi_graph_cnn_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes:]Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix
î
j
;__inference_z_layer_call_and_return_conditional_losses_3979
inputs_0
inputs_1
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÌÜæ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
©

&__inference_dense_1_layer_call_fn_3900

inputs 
dense_1_kernel:
dense_1_bias:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

__inference__wrapped_model_2609
node_attributes
adjacency_matrixW
Eencoder_multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel:dQ
Cencoder_multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias:d\
Iencoder_multi_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel:	ÈdU
Gencoder_multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias:dB
0encoder_dense_matmul_readvariableop_dense_kernel:d=
/encoder_dense_biasadd_readvariableop_dense_bias:F
4encoder_dense_1_matmul_readvariableop_dense_1_kernel:A
3encoder_dense_1_biasadd_readvariableop_dense_1_bias:D
2encoder_z_mean_matmul_readvariableop_z_mean_kernel:?
1encoder_z_mean_biasadd_readvariableop_z_mean_bias:J
8encoder_z_log_var_matmul_readvariableop_z_log_var_kernel:E
7encoder_z_log_var_biasadd_readvariableop_z_log_var_bias:
identity

identity_1

identity_2¢$encoder/dense/BiasAdd/ReadVariableOp¢#encoder/dense/MatMul/ReadVariableOp¢&encoder/dense_1/BiasAdd/ReadVariableOp¢%encoder/dense_1/MatMul/ReadVariableOp¢.encoder/multi_graph_cnn/BiasAdd/ReadVariableOp¢0encoder/multi_graph_cnn/transpose/ReadVariableOp¢0encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOp¢2encoder/multi_graph_cnn_1/transpose/ReadVariableOp¢(encoder/z_log_var/BiasAdd/ReadVariableOp¢'encoder/z_log_var/MatMul/ReadVariableOp¢%encoder/z_mean/BiasAdd/ReadVariableOp¢$encoder/z_mean/MatMul/ReadVariableOp
encoder/multi_graph_cnn/MatMulBatchMatMulV2adjacency_matrixnode_attributes*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
encoder/multi_graph_cnn/ShapeShape'encoder/multi_graph_cnn/MatMul:output:0*
T0*
_output_shapes
:i
'encoder/multi_graph_cnn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ß
encoder/multi_graph_cnn/splitSplit0encoder/multi_graph_cnn/split/split_dim:output:0'encoder/multi_graph_cnn/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splite
#encoder/multi_graph_cnn/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ç
encoder/multi_graph_cnn/concatConcatV2&encoder/multi_graph_cnn/split:output:0&encoder/multi_graph_cnn/split:output:1,encoder/multi_graph_cnn/concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
encoder/multi_graph_cnn/Shape_1Shape'encoder/multi_graph_cnn/concat:output:0*
T0*
_output_shapes
:
encoder/multi_graph_cnn/unstackUnpack(encoder/multi_graph_cnn/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num´
.encoder/multi_graph_cnn/Shape_2/ReadVariableOpReadVariableOpEencoder_multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0p
encoder/multi_graph_cnn/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   d   
!encoder/multi_graph_cnn/unstack_1Unpack(encoder/multi_graph_cnn/Shape_2:output:0*
T0*
_output_shapes
: : *	
numv
%encoder/multi_graph_cnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   µ
encoder/multi_graph_cnn/ReshapeReshape'encoder/multi_graph_cnn/concat:output:0.encoder/multi_graph_cnn/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
0encoder/multi_graph_cnn/transpose/ReadVariableOpReadVariableOpEencoder_multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0w
&encoder/multi_graph_cnn/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Â
!encoder/multi_graph_cnn/transpose	Transpose8encoder/multi_graph_cnn/transpose/ReadVariableOp:value:0/encoder/multi_graph_cnn/transpose/perm:output:0*
T0*
_output_shapes

:dx
'encoder/multi_graph_cnn/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ®
!encoder/multi_graph_cnn/Reshape_1Reshape%encoder/multi_graph_cnn/transpose:y:00encoder/multi_graph_cnn/Reshape_1/shape:output:0*
T0*
_output_shapes

:d²
 encoder/multi_graph_cnn/MatMul_1MatMul(encoder/multi_graph_cnn/Reshape:output:0*encoder/multi_graph_cnn/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdk
)encoder/multi_graph_cnn/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :k
)encoder/multi_graph_cnn/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :dï
'encoder/multi_graph_cnn/Reshape_2/shapePack(encoder/multi_graph_cnn/unstack:output:02encoder/multi_graph_cnn/Reshape_2/shape/1:output:02encoder/multi_graph_cnn/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:À
!encoder/multi_graph_cnn/Reshape_2Reshape*encoder/multi_graph_cnn/MatMul_1:product:00encoder/multi_graph_cnn/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd®
.encoder/multi_graph_cnn/BiasAdd/ReadVariableOpReadVariableOpCencoder_multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias*
_output_shapes
:d*
dtype0Ä
encoder/multi_graph_cnn/BiasAddBiasAdd*encoder/multi_graph_cnn/Reshape_2:output:06encoder/multi_graph_cnn/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
encoder/multi_graph_cnn/EluElu(encoder/multi_graph_cnn/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
encoder/dropout/IdentityIdentity)encoder/multi_graph_cnn/Elu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 encoder/multi_graph_cnn_1/MatMulBatchMatMulV2adjacency_matrix!encoder/dropout/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
encoder/multi_graph_cnn_1/ShapeShape)encoder/multi_graph_cnn_1/MatMul:output:0*
T0*
_output_shapes
:k
)encoder/multi_graph_cnn_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :å
encoder/multi_graph_cnn_1/splitSplit2encoder/multi_graph_cnn_1/split/split_dim:output:0)encoder/multi_graph_cnn_1/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitg
%encoder/multi_graph_cnn_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ð
 encoder/multi_graph_cnn_1/concatConcatV2(encoder/multi_graph_cnn_1/split:output:0(encoder/multi_graph_cnn_1/split:output:1.encoder/multi_graph_cnn_1/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈz
!encoder/multi_graph_cnn_1/Shape_1Shape)encoder/multi_graph_cnn_1/concat:output:0*
T0*
_output_shapes
:
!encoder/multi_graph_cnn_1/unstackUnpack*encoder/multi_graph_cnn_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num»
0encoder/multi_graph_cnn_1/Shape_2/ReadVariableOpReadVariableOpIencoder_multi_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0r
!encoder/multi_graph_cnn_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"È   d   
#encoder/multi_graph_cnn_1/unstack_1Unpack*encoder/multi_graph_cnn_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
numx
'encoder/multi_graph_cnn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   ¼
!encoder/multi_graph_cnn_1/ReshapeReshape)encoder/multi_graph_cnn_1/concat:output:00encoder/multi_graph_cnn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ½
2encoder/multi_graph_cnn_1/transpose/ReadVariableOpReadVariableOpIencoder_multi_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0y
(encoder/multi_graph_cnn_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       É
#encoder/multi_graph_cnn_1/transpose	Transpose:encoder/multi_graph_cnn_1/transpose/ReadVariableOp:value:01encoder/multi_graph_cnn_1/transpose/perm:output:0*
T0*
_output_shapes
:	Èdz
)encoder/multi_graph_cnn_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"È   ÿÿÿÿµ
#encoder/multi_graph_cnn_1/Reshape_1Reshape'encoder/multi_graph_cnn_1/transpose:y:02encoder/multi_graph_cnn_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Èd¸
"encoder/multi_graph_cnn_1/MatMul_1MatMul*encoder/multi_graph_cnn_1/Reshape:output:0,encoder/multi_graph_cnn_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
+encoder/multi_graph_cnn_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :m
+encoder/multi_graph_cnn_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d÷
)encoder/multi_graph_cnn_1/Reshape_2/shapePack*encoder/multi_graph_cnn_1/unstack:output:04encoder/multi_graph_cnn_1/Reshape_2/shape/1:output:04encoder/multi_graph_cnn_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Æ
#encoder/multi_graph_cnn_1/Reshape_2Reshape,encoder/multi_graph_cnn_1/MatMul_1:product:02encoder/multi_graph_cnn_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
0encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOpReadVariableOpGencoder_multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias*
_output_shapes
:d*
dtype0Ê
!encoder/multi_graph_cnn_1/BiasAddBiasAdd,encoder/multi_graph_cnn_1/Reshape_2:output:08encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
encoder/multi_graph_cnn_1/EluElu*encoder/multi_graph_cnn_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
encoder/dropout_1/IdentityIdentity+encoder/multi_graph_cnn_1/Elu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdg
%encoder/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¢
encoder/lambda/MeanMean#encoder/dropout_1/Identity:output:0.encoder/lambda/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#encoder/dense/MatMul/ReadVariableOpReadVariableOp0encoder_dense_matmul_readvariableop_dense_kernel*
_output_shapes

:d*
dtype0
encoder/dense/MatMulMatMulencoder/lambda/Mean:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0 
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp4encoder_dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:*
dtype0£
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp3encoder_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0¦
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
encoder/dense_1/ReluRelu encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp2encoder_z_mean_matmul_readvariableop_z_mean_kernel*
_output_shapes

:*
dtype0£
encoder/z_mean/MatMulMatMul"encoder/dense_1/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_mean_biasadd_readvariableop_z_mean_bias*
_output_shapes
:*
dtype0£
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp8encoder_z_log_var_matmul_readvariableop_z_log_var_kernel*
_output_shapes

:*
dtype0©
encoder/z_log_var/MatMulMatMul"encoder/dense_1/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp7encoder_z_log_var_biasadd_readvariableop_z_log_var_bias*
_output_shapes
:*
dtype0¬
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
encoder/z/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:g
encoder/z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
encoder/z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
encoder/z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
encoder/z/strided_sliceStridedSliceencoder/z/Shape:output:0&encoder/z/strided_slice/stack:output:0(encoder/z/strided_slice/stack_1:output:0(encoder/z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
encoder/z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
encoder/z/random_normal/shapePack encoder/z/strided_slice:output:0(encoder/z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:a
encoder/z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    c
encoder/z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ê
,encoder/z/random_normal/RandomStandardNormalRandomStandardNormal&encoder/z/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ËÖÇ´
encoder/z/random_normal/mulMul5encoder/z/random_normal/RandomStandardNormal:output:0'encoder/z/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
encoder/z/random_normalAddV2encoder/z/random_normal/mul:z:0%encoder/z/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
encoder/z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
encoder/z/mulMulencoder/z/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
encoder/z/ExpExpencoder/z/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
encoder/z/mul_1Mulencoder/z/Exp:y:0encoder/z/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
encoder/z/addAddV2encoder/z_mean/BiasAdd:output:0encoder/z/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityencoder/z/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp/^encoder/multi_graph_cnn/BiasAdd/ReadVariableOp1^encoder/multi_graph_cnn/transpose/ReadVariableOp1^encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOp3^encoder/multi_graph_cnn_1/transpose/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp2`
.encoder/multi_graph_cnn/BiasAdd/ReadVariableOp.encoder/multi_graph_cnn/BiasAdd/ReadVariableOp2d
0encoder/multi_graph_cnn/transpose/ReadVariableOp0encoder/multi_graph_cnn/transpose/ReadVariableOp2d
0encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOp0encoder/multi_graph_cnn_1/BiasAdd/ReadVariableOp2h
2encoder/multi_graph_cnn_1/transpose/ReadVariableOp2encoder/multi_graph_cnn_1/transpose/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes:]Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix
æ
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_2712

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Â
\
@__inference_lambda_layer_call_and_return_conditional_losses_2993

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö
A
%__inference_lambda_layer_call_fn_3863

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2993`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs

µ
&__inference_encoder_layer_call_fn_2826
node_attributes
adjacency_matrix(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
dense_kernel:d

dense_bias: 
dense_1_kernel:
dense_1_bias:
z_mean_kernel:
z_mean_bias:"
z_log_var_kernel:
z_log_var_bias:
identity

identity_1

identity_2¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallnode_attributesadjacency_matrixmulti_graph_cnn_kernelmulti_graph_cnn_biasmulti_graph_cnn_1_kernelmulti_graph_cnn_1_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasz_mean_kernelz_mean_biasz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_2807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes:]Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix
Â
\
@__inference_lambda_layer_call_and_return_conditional_losses_2720

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ö
a
(__inference_dropout_1_layer_call_fn_3836

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_3028s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
î
j
;__inference_z_layer_call_and_return_conditional_losses_4001
inputs_0
inputs_1
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÿ¥¿
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
3
¾
 __inference__traced_restore_4109
file_prefix9
'assignvariableop_multi_graph_cnn_kernel:d5
'assignvariableop_1_multi_graph_cnn_bias:d>
+assignvariableop_2_multi_graph_cnn_1_kernel:	Èd7
)assignvariableop_3_multi_graph_cnn_1_bias:d1
assignvariableop_4_dense_kernel:d+
assignvariableop_5_dense_bias:3
!assignvariableop_6_dense_1_kernel:-
assignvariableop_7_dense_1_bias:2
 assignvariableop_8_z_mean_kernel:,
assignvariableop_9_z_mean_bias:6
$assignvariableop_10_z_log_var_kernel:0
"assignvariableop_11_z_log_var_bias:
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¡
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ß
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp'assignvariableop_multi_graph_cnn_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp'assignvariableop_1_multi_graph_cnn_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp+assignvariableop_2_multi_graph_cnn_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp)assignvariableop_3_multi_graph_cnn_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp assignvariableop_8_z_mean_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_z_mean_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_z_log_var_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp"assignvariableop_11_z_log_var_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
3
 
A__inference_encoder_layer_call_and_return_conditional_losses_2807

inputs
inputs_18
&multi_graph_cnn_multi_graph_cnn_kernel:d2
$multi_graph_cnn_multi_graph_cnn_bias:d=
*multi_graph_cnn_1_multi_graph_cnn_1_kernel:	Èd6
(multi_graph_cnn_1_multi_graph_cnn_1_bias:d$
dense_dense_kernel:d
dense_dense_bias:(
dense_1_dense_1_kernel:"
dense_1_dense_1_bias:&
z_mean_z_mean_kernel: 
z_mean_z_mean_bias:,
z_log_var_z_log_var_kernel:&
z_log_var_z_log_var_bias:
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢'multi_graph_cnn/StatefulPartitionedCall¢)multi_graph_cnn_1/StatefulPartitionedCall¢z/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCallº
'multi_graph_cnn/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1&multi_graph_cnn_multi_graph_cnn_kernel$multi_graph_cnn_multi_graph_cnn_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655å
dropout/PartitionedCallPartitionedCall0multi_graph_cnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2664à
)multi_graph_cnn_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0inputs_1*multi_graph_cnn_1_multi_graph_cnn_1_kernel(multi_graph_cnn_1_multi_graph_cnn_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703ë
dropout_1/PartitionedCallPartitionedCall2multi_graph_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2712Ñ
lambda/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2720
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2733
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2748
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_z_mean_kernelz_mean_z_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_2762©
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_z_log_var_kernelz_log_var_z_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_2776
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2802v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^multi_graph_cnn/StatefulPartitionedCall*^multi_graph_cnn_1/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'multi_graph_cnn/StatefulPartitionedCall'multi_graph_cnn/StatefulPartitionedCall2V
)multi_graph_cnn_1/StatefulPartitionedCall)multi_graph_cnn_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

\
@__inference_lambda_layer_call_and_return_conditional_losses_3875

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
©	
ù
@__inference_z_mean_layer_call_and_return_conditional_losses_3928

inputs5
#matmul_readvariableop_z_mean_kernel:0
"biasadd_readvariableop_z_mean_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp#matmul_readvariableop_z_mean_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
BiasAdd/ReadVariableOpReadVariableOp"biasadd_readvariableop_z_mean_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

%__inference_z_mean_layer_call_fn_3918

inputs
z_mean_kernel:
z_mean_bias:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsz_mean_kernelz_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_2762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

µ
&__inference_encoder_layer_call_fn_3333
node_attributes
adjacency_matrix(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
dense_kernel:d

dense_bias: 
dense_1_kernel:
dense_1_bias:
z_mean_kernel:
z_mean_bias:"
z_log_var_kernel:
z_log_var_bias:
identity

identity_1

identity_2¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallnode_attributesadjacency_matrixmulti_graph_cnn_kernelmulti_graph_cnn_biasmulti_graph_cnn_1_kernelmulti_graph_cnn_1_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasz_mean_kernelz_mean_biasz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_3234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes:]Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix
à
¦
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655

inputs
inputs_1?
-shape_2_readvariableop_multi_graph_cnn_kernel:d9
+biasadd_readvariableop_multi_graph_cnn_bias:d
identity¢BiasAdd/ReadVariableOp¢transpose/ReadVariableOp_
MatMulBatchMatMulV2inputs_1inputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:1concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
Shape_1Shapeconcat:output:0*
T0*
_output_shapes
:S
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num
Shape_2/ReadVariableOpReadVariableOp-shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   d   S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   m
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
transpose/ReadVariableOpReadVariableOp-shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:d`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿf
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:dj
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd~
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_multi_graph_cnn_bias*
_output_shapes
:d*
dtype0|
BiasAddBiasAddReshape_2:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
NoOpNoOp^BiasAdd/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

\
@__inference_lambda_layer_call_and_return_conditional_losses_3869

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
â	

C__inference_z_log_var_layer_call_and_return_conditional_losses_2776

inputs8
&matmul_readvariableop_z_log_var_kernel:3
%biasadd_readvariableop_z_log_var_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp&matmul_readvariableop_z_log_var_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
BiasAdd/ReadVariableOpReadVariableOp%biasadd_readvariableop_z_log_var_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿#
´
__inference__traced_save_4063
file_prefix5
1savev2_multi_graph_cnn_kernel_read_readvariableop3
/savev2_multi_graph_cnn_bias_read_readvariableop7
3savev2_multi_graph_cnn_1_kernel_read_readvariableop5
1savev2_multi_graph_cnn_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B Ö
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_multi_graph_cnn_kernel_read_readvariableop/savev2_multi_graph_cnn_bias_read_readvariableop3savev2_multi_graph_cnn_1_kernel_read_readvariableop1savev2_multi_graph_cnn_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*x
_input_shapesg
e: :d:d:	Èd:d:d:::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:%!

_output_shapes
:	Èd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
ø	
ö
?__inference_dense_layer_call_and_return_conditional_losses_3893

inputs4
"matmul_readvariableop_dense_kernel:d/
!biasadd_readvariableop_dense_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpx
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs


`
A__inference_dropout_layer_call_and_return_conditional_losses_3114

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
­3
±
A__inference_encoder_layer_call_and_return_conditional_losses_3362
node_attributes
adjacency_matrix8
&multi_graph_cnn_multi_graph_cnn_kernel:d2
$multi_graph_cnn_multi_graph_cnn_bias:d=
*multi_graph_cnn_1_multi_graph_cnn_1_kernel:	Èd6
(multi_graph_cnn_1_multi_graph_cnn_1_bias:d$
dense_dense_kernel:d
dense_dense_bias:(
dense_1_dense_1_kernel:"
dense_1_dense_1_bias:&
z_mean_z_mean_kernel: 
z_mean_z_mean_bias:,
z_log_var_z_log_var_kernel:&
z_log_var_z_log_var_bias:
identity

identity_1

identity_2¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢'multi_graph_cnn/StatefulPartitionedCall¢)multi_graph_cnn_1/StatefulPartitionedCall¢z/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCallË
'multi_graph_cnn/StatefulPartitionedCallStatefulPartitionedCallnode_attributesadjacency_matrix&multi_graph_cnn_multi_graph_cnn_kernel$multi_graph_cnn_multi_graph_cnn_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_2655å
dropout/PartitionedCallPartitionedCall0multi_graph_cnn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2664è
)multi_graph_cnn_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0adjacency_matrix*multi_graph_cnn_1_multi_graph_cnn_1_kernel(multi_graph_cnn_1_multi_graph_cnn_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703ë
dropout_1/PartitionedCallPartitionedCall2multi_graph_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_2712Ñ
lambda/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2720
dense/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2733
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_2748
z_mean/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_mean_z_mean_kernelz_mean_z_mean_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_z_mean_layer_call_and_return_conditional_losses_2762©
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0z_log_var_z_log_var_kernelz_log_var_z_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_z_log_var_layer_call_and_return_conditional_losses_2776
z/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2802v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs

Identity_2Identity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall(^multi_graph_cnn/StatefulPartitionedCall*^multi_graph_cnn_1/StatefulPartitionedCall^z/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2R
'multi_graph_cnn/StatefulPartitionedCall'multi_graph_cnn/StatefulPartitionedCall2V
)multi_graph_cnn_1/StatefulPartitionedCall)multi_graph_cnn_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes:]Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix

h
;__inference_z_layer_call_and_return_conditional_losses_2858

inputs
inputs_1
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ËÞ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
i
 __inference_z_layer_call_fn_3951
inputs_0
inputs_1
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *D
f?R=
;__inference_z_layer_call_and_return_conditional_losses_2802o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
	
Ì
0__inference_multi_graph_cnn_1_layer_call_fn_3789
inputs_0
inputs_1+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1multi_graph_cnn_1_kernelmulti_graph_cnn_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_2703s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
å	
`
A__inference_dropout_layer_call_and_return_conditional_losses_3781

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ö
A
%__inference_lambda_layer_call_fn_3858

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_lambda_layer_call_and_return_conditional_losses_2720`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ò
±
"__inference_signature_wrapper_3709
adjacency_matrix
node_attributes(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
dense_kernel:d

dense_bias: 
dense_1_kernel:
dense_1_bias:
z_mean_kernel:
z_mean_bias:"
z_log_var_kernel:
z_log_var_bias:
identity

identity_1

identity_2¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallnode_attributesadjacency_matrixmulti_graph_cnn_kernelmulti_graph_cnn_biasmulti_graph_cnn_1_kernelmulti_graph_cnn_1_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasz_mean_kernelz_mean_biasz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_2609o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameadjacency_matrix:\X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namenode_attributes
ë
¦
&__inference_encoder_layer_call_fn_3413
inputs_0
inputs_1(
multi_graph_cnn_kernel:d"
multi_graph_cnn_bias:d+
multi_graph_cnn_1_kernel:	Èd$
multi_graph_cnn_1_bias:d
dense_kernel:d

dense_bias: 
dense_1_kernel:
dense_1_bias:
z_mean_kernel:
z_mean_bias:"
z_log_var_kernel:
z_log_var_bias:
identity

identity_1

identity_2¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1multi_graph_cnn_kernelmulti_graph_cnn_biasmulti_graph_cnn_1_kernelmulti_graph_cnn_1_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasz_mean_kernelz_mean_biasz_log_var_kernelz_log_var_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_2807o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


$__inference_dense_layer_call_fn_3882

inputs
dense_kernel:d

dense_bias:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_2733o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ä
_
A__inference_dropout_layer_call_and_return_conditional_losses_2664

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ä
ç

A__inference_encoder_layer_call_and_return_conditional_losses_3685
inputs_0
inputs_1O
=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel:dI
;multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias:dT
Amulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel:	ÈdM
?multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias:d:
(dense_matmul_readvariableop_dense_kernel:d5
'dense_biasadd_readvariableop_dense_bias:>
,dense_1_matmul_readvariableop_dense_1_kernel:9
+dense_1_biasadd_readvariableop_dense_1_bias:<
*z_mean_matmul_readvariableop_z_mean_kernel:7
)z_mean_biasadd_readvariableop_z_mean_bias:B
0z_log_var_matmul_readvariableop_z_log_var_kernel:=
/z_log_var_biasadd_readvariableop_z_log_var_bias:
identity

identity_1

identity_2¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢&multi_graph_cnn/BiasAdd/ReadVariableOp¢(multi_graph_cnn/transpose/ReadVariableOp¢(multi_graph_cnn_1/BiasAdd/ReadVariableOp¢*multi_graph_cnn_1/transpose/ReadVariableOp¢ z_log_var/BiasAdd/ReadVariableOp¢z_log_var/MatMul/ReadVariableOp¢z_mean/BiasAdd/ReadVariableOp¢z_mean/MatMul/ReadVariableOpq
multi_graph_cnn/MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
multi_graph_cnn/ShapeShapemulti_graph_cnn/MatMul:output:0*
T0*
_output_shapes
:a
multi_graph_cnn/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ç
multi_graph_cnn/splitSplit(multi_graph_cnn/split/split_dim:output:0multi_graph_cnn/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split]
multi_graph_cnn/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
multi_graph_cnn/concatConcatV2multi_graph_cnn/split:output:0multi_graph_cnn/split:output:1$multi_graph_cnn/concat/axis:output:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
multi_graph_cnn/Shape_1Shapemulti_graph_cnn/concat:output:0*
T0*
_output_shapes
:s
multi_graph_cnn/unstackUnpack multi_graph_cnn/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num¤
&multi_graph_cnn/Shape_2/ReadVariableOpReadVariableOp=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0h
multi_graph_cnn/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   d   s
multi_graph_cnn/unstack_1Unpack multi_graph_cnn/Shape_2:output:0*
T0*
_output_shapes
: : *	
numn
multi_graph_cnn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
multi_graph_cnn/ReshapeReshapemulti_graph_cnn/concat:output:0&multi_graph_cnn/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
(multi_graph_cnn/transpose/ReadVariableOpReadVariableOp=multi_graph_cnn_shape_2_readvariableop_multi_graph_cnn_kernel*
_output_shapes

:d*
dtype0o
multi_graph_cnn/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ª
multi_graph_cnn/transpose	Transpose0multi_graph_cnn/transpose/ReadVariableOp:value:0'multi_graph_cnn/transpose/perm:output:0*
T0*
_output_shapes

:dp
multi_graph_cnn/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ
multi_graph_cnn/Reshape_1Reshapemulti_graph_cnn/transpose:y:0(multi_graph_cnn/Reshape_1/shape:output:0*
T0*
_output_shapes

:d
multi_graph_cnn/MatMul_1MatMul multi_graph_cnn/Reshape:output:0"multi_graph_cnn/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
!multi_graph_cnn/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!multi_graph_cnn/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :dÏ
multi_graph_cnn/Reshape_2/shapePack multi_graph_cnn/unstack:output:0*multi_graph_cnn/Reshape_2/shape/1:output:0*multi_graph_cnn/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:¨
multi_graph_cnn/Reshape_2Reshape"multi_graph_cnn/MatMul_1:product:0(multi_graph_cnn/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&multi_graph_cnn/BiasAdd/ReadVariableOpReadVariableOp;multi_graph_cnn_biasadd_readvariableop_multi_graph_cnn_bias*
_output_shapes
:d*
dtype0¬
multi_graph_cnn/BiasAddBiasAdd"multi_graph_cnn/Reshape_2:output:0.multi_graph_cnn/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
multi_graph_cnn/EluElu multi_graph_cnn/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout/dropout/MulMul!multi_graph_cnn/Elu:activations:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
dropout/dropout/ShapeShape!multi_graph_cnn/Elu:activations:0*
T0*
_output_shapes
: 
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Â
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
multi_graph_cnn_1/MatMulBatchMatMulV2inputs_1dropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdh
multi_graph_cnn_1/ShapeShape!multi_graph_cnn_1/MatMul:output:0*
T0*
_output_shapes
:c
!multi_graph_cnn_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Í
multi_graph_cnn_1/splitSplit*multi_graph_cnn_1/split/split_dim:output:0!multi_graph_cnn_1/MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split_
multi_graph_cnn_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ð
multi_graph_cnn_1/concatConcatV2 multi_graph_cnn_1/split:output:0 multi_graph_cnn_1/split:output:1&multi_graph_cnn_1/concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈj
multi_graph_cnn_1/Shape_1Shape!multi_graph_cnn_1/concat:output:0*
T0*
_output_shapes
:w
multi_graph_cnn_1/unstackUnpack"multi_graph_cnn_1/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num«
(multi_graph_cnn_1/Shape_2/ReadVariableOpReadVariableOpAmulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0j
multi_graph_cnn_1/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"È   d   w
multi_graph_cnn_1/unstack_1Unpack"multi_graph_cnn_1/Shape_2:output:0*
T0*
_output_shapes
: : *	
nump
multi_graph_cnn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   ¤
multi_graph_cnn_1/ReshapeReshape!multi_graph_cnn_1/concat:output:0(multi_graph_cnn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ­
*multi_graph_cnn_1/transpose/ReadVariableOpReadVariableOpAmulti_graph_cnn_1_shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0q
 multi_graph_cnn_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ±
multi_graph_cnn_1/transpose	Transpose2multi_graph_cnn_1/transpose/ReadVariableOp:value:0)multi_graph_cnn_1/transpose/perm:output:0*
T0*
_output_shapes
:	Èdr
!multi_graph_cnn_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"È   ÿÿÿÿ
multi_graph_cnn_1/Reshape_1Reshapemulti_graph_cnn_1/transpose:y:0*multi_graph_cnn_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Èd 
multi_graph_cnn_1/MatMul_1MatMul"multi_graph_cnn_1/Reshape:output:0$multi_graph_cnn_1/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿde
#multi_graph_cnn_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#multi_graph_cnn_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d×
!multi_graph_cnn_1/Reshape_2/shapePack"multi_graph_cnn_1/unstack:output:0,multi_graph_cnn_1/Reshape_2/shape/1:output:0,multi_graph_cnn_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:®
multi_graph_cnn_1/Reshape_2Reshape$multi_graph_cnn_1/MatMul_1:product:0*multi_graph_cnn_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¤
(multi_graph_cnn_1/BiasAdd/ReadVariableOpReadVariableOp?multi_graph_cnn_1_biasadd_readvariableop_multi_graph_cnn_1_bias*
_output_shapes
:d*
dtype0²
multi_graph_cnn_1/BiasAddBiasAdd$multi_graph_cnn_1/Reshape_2:output:00multi_graph_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdv
multi_graph_cnn_1/EluElu"multi_graph_cnn_1/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMul#multi_graph_cnn_1/Elu:activations:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
dropout_1/dropout/ShapeShape#multi_graph_cnn_1/Elu:activations:0*
T0*
_output_shapes
:¤
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
lambda/MeanMeandropout_1/dropout/Mul_1:z:0&lambda/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes

:d*
dtype0
dense/MatMulMatMullambda/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel*
_output_shapes

:*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_mean/MatMul/ReadVariableOpReadVariableOp*z_mean_matmul_readvariableop_z_mean_kernel*
_output_shapes

:*
dtype0
z_mean/MatMulMatMuldense_1/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_mean/BiasAdd/ReadVariableOpReadVariableOp)z_mean_biasadd_readvariableop_z_mean_bias*
_output_shapes
:*
dtype0
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z_log_var/MatMul/ReadVariableOpReadVariableOp0z_log_var_matmul_readvariableop_z_log_var_kernel*
_output_shapes

:*
dtype0
z_log_var/MatMulMatMuldense_1/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp/z_log_var_biasadd_readvariableop_z_log_var_bias*
_output_shapes
:*
dtype0
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
z/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:_
z/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: a
z/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
z/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
z/strided_sliceStridedSlicez/Shape:output:0z/strided_slice/stack:output:0 z/strided_slice/stack_1:output:0 z/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
z/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
z/random_normal/shapePackz/strided_slice:output:0 z/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:Y
z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    [
z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
$z/random_normal/RandomStandardNormalRandomStandardNormalz/random_normal/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Õ¼
z/random_normal/mulMul-z/random_normal/RandomStandardNormal:output:0z/random_normal/stddev:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z/random_normalAddV2z/random_normal/mul:z:0z/random_normal/mean:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
z/mulMulz/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
z/ExpExp	z/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
z/mul_1Mul	z/Exp:y:0z/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
z/addAddV2z_mean/BiasAdd:output:0z/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	z/add:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp'^multi_graph_cnn/BiasAdd/ReadVariableOp)^multi_graph_cnn/transpose/ReadVariableOp)^multi_graph_cnn_1/BiasAdd/ReadVariableOp+^multi_graph_cnn_1/transpose/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Y
_input_shapesH
F:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2P
&multi_graph_cnn/BiasAdd/ReadVariableOp&multi_graph_cnn/BiasAdd/ReadVariableOp2T
(multi_graph_cnn/transpose/ReadVariableOp(multi_graph_cnn/transpose/ReadVariableOp2T
(multi_graph_cnn_1/BiasAdd/ReadVariableOp(multi_graph_cnn_1/BiasAdd/ReadVariableOp2X
*multi_graph_cnn_1/transpose/ReadVariableOp*multi_graph_cnn_1/transpose/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


ü
A__inference_dense_1_layer_call_and_return_conditional_losses_3911

inputs6
$matmul_readvariableop_dense_1_kernel:1
#biasadd_readvariableop_dense_1_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç	
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_3853

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿds
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Ò
¯
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_3826
inputs_0
inputs_1B
/shape_2_readvariableop_multi_graph_cnn_1_kernel:	Èd;
-biasadd_readvariableop_multi_graph_cnn_1_bias:d
identity¢BiasAdd/ReadVariableOp¢transpose/ReadVariableOpa
MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0MatMul:output:0*
T0*B
_output_shapes0
.:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2split:output:0split:output:1concat/axis:output:0*
N*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈF
Shape_1Shapeconcat:output:0*
T0*
_output_shapes
:S
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num
Shape_2/ReadVariableOpReadVariableOp/shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"È   d   S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   n
ReshapeReshapeconcat:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
transpose/ReadVariableOpReadVariableOp/shape_2_readvariableop_multi_graph_cnn_1_kernel*
_output_shapes
:	Èd*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes
:	Èd`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"È   ÿÿÿÿg
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	Èdj
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdS
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_multi_graph_cnn_1_bias*
_output_shapes
:d*
dtype0|
BiasAddBiasAddReshape_2:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
EluEluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
IdentityIdentityElu:activations:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
NoOpNoOp^BiasAdd/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultò
Q
adjacency_matrix=
"serving_default_adjacency_matrix:0ÿÿÿÿÿÿÿÿÿ
O
node_attributes<
!serving_default_node_attributes:0ÿÿÿÿÿÿÿÿÿ5
z0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ=
	z_log_var0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿ:
z_mean0
StatefulPartitionedCall:2ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¹

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
	variables
trainable_variables
regularization_losses
 	keras_api
!_random_generator
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
»

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
v
0
1
$2
%3
94
:5
A6
B7
I8
J9
Q10
R11"
trackable_list_wrapper
v
0
1
$2
%3
94
:5
A6
B7
I8
J9
Q10
R11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ2ã
&__inference_encoder_layer_call_fn_2826
&__inference_encoder_layer_call_fn_3413
&__inference_encoder_layer_call_fn_3435
&__inference_encoder_layer_call_fn_3333À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_encoder_layer_call_and_return_conditional_losses_3553
A__inference_encoder_layer_call_and_return_conditional_losses_3685
A__inference_encoder_layer_call_and_return_conditional_losses_3362
A__inference_encoder_layer_call_and_return_conditional_losses_3391À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
äBá
__inference__wrapped_model_2609node_attributesadjacency_matrix"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
dserving_default"
signature_map
(:&d2multi_graph_cnn/kernel
": d2multi_graph_cnn/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_multi_graph_cnn_layer_call_fn_3717¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_3754¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
&__inference_dropout_layer_call_fn_3759
&__inference_dropout_layer_call_fn_3764´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_3769
A__inference_dropout_layer_call_and_return_conditional_losses_3781´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
+:)	Èd2multi_graph_cnn_1/kernel
$:"d2multi_graph_cnn_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_multi_graph_cnn_1_layer_call_fn_3789¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_3826¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
(__inference_dropout_1_layer_call_fn_3831
(__inference_dropout_1_layer_call_fn_3836´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á
C__inference_dropout_1_layer_call_and_return_conditional_losses_3841
C__inference_dropout_1_layer_call_and_return_conditional_losses_3853´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
2
%__inference_lambda_layer_call_fn_3858
%__inference_lambda_layer_call_fn_3863À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
@__inference_lambda_layer_call_and_return_conditional_losses_3869
@__inference_lambda_layer_call_and_return_conditional_losses_3875À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:d2dense/kernel
:2
dense/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dense_layer_call_fn_3882¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_3893¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_1/kernel
:2dense_1/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dense_1_layer_call_fn_3900¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_3911¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:2z_mean/kernel
:2z_mean/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_z_mean_layer_call_fn_3918¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_z_mean_layer_call_and_return_conditional_losses_3928¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 2z_log_var/kernel
:2z_log_var/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_z_log_var_layer_call_fn_3935¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_z_log_var_layer_call_and_return_conditional_losses_3945¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
2
 __inference_z_layer_call_fn_3951
 __inference_z_layer_call_fn_3957À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
;__inference_z_layer_call_and_return_conditional_losses_3979
;__inference_z_layer_call_and_return_conditional_losses_4001À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
"__inference_signature_wrapper_3709adjacency_matrixnode_attributes"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper¬
__inference__wrapped_model_2609$%9:ABIJQRq¢n
g¢d
b_
-*
node_attributesÿÿÿÿÿÿÿÿÿ
.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
ª "ª
 
z
zÿÿÿÿÿÿÿÿÿ
0
	z_log_var# 
	z_log_varÿÿÿÿÿÿÿÿÿ
*
z_mean 
z_meanÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_1_layer_call_and_return_conditional_losses_3911\AB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_1_layer_call_fn_3900OAB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
?__inference_dense_layer_call_and_return_conditional_losses_3893\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dense_layer_call_fn_3882O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ«
C__inference_dropout_1_layer_call_and_return_conditional_losses_3841d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 «
C__inference_dropout_1_layer_call_and_return_conditional_losses_3853d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 
(__inference_dropout_1_layer_call_fn_3831W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd
(__inference_dropout_1_layer_call_fn_3836W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd©
A__inference_dropout_layer_call_and_return_conditional_losses_3769d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 ©
A__inference_dropout_layer_call_and_return_conditional_losses_3781d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 
&__inference_dropout_layer_call_fn_3759W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p 
ª "ÿÿÿÿÿÿÿÿÿd
&__inference_dropout_layer_call_fn_3764W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿd
p
ª "ÿÿÿÿÿÿÿÿÿd»
A__inference_encoder_layer_call_and_return_conditional_losses_3362õ$%9:ABIJQRy¢v
o¢l
b_
-*
node_attributesÿÿÿÿÿÿÿÿÿ
.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 »
A__inference_encoder_layer_call_and_return_conditional_losses_3391õ$%9:ABIJQRy¢v
o¢l
b_
-*
node_attributesÿÿÿÿÿÿÿÿÿ
.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ¬
A__inference_encoder_layer_call_and_return_conditional_losses_3553æ$%9:ABIJQRj¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ¬
A__inference_encoder_layer_call_and_return_conditional_losses_3685æ$%9:ABIJQRj¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
&__inference_encoder_layer_call_fn_2826å$%9:ABIJQRy¢v
o¢l
b_
-*
node_attributesÿÿÿÿÿÿÿÿÿ
.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ
&__inference_encoder_layer_call_fn_3333å$%9:ABIJQRy¢v
o¢l
b_
-*
node_attributesÿÿÿÿÿÿÿÿÿ
.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ
&__inference_encoder_layer_call_fn_3413Ö$%9:ABIJQRj¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ
&__inference_encoder_layer_call_fn_3435Ö$%9:ABIJQRj¢g
`¢]
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ¨
@__inference_lambda_layer_call_and_return_conditional_losses_3869d;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ¨
@__inference_lambda_layer_call_and_return_conditional_losses_3875d;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
%__inference_lambda_layer_call_fn_3858W;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p 
ª "ÿÿÿÿÿÿÿÿÿd
%__inference_lambda_layer_call_fn_3863W;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿd

 
p
ª "ÿÿÿÿÿÿÿÿÿdã
K__inference_multi_graph_cnn_1_layer_call_and_return_conditional_losses_3826$%b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿd
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 »
0__inference_multi_graph_cnn_1_layer_call_fn_3789$%b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿd
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿdá
I__inference_multi_graph_cnn_layer_call_and_return_conditional_losses_3754b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd
 ¹
.__inference_multi_graph_cnn_layer_call_fn_3717b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
&#
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿdÕ
"__inference_signature_wrapper_3709®$%9:ABIJQR¢
¢ 
ª
B
adjacency_matrix.+
adjacency_matrixÿÿÿÿÿÿÿÿÿ
@
node_attributes-*
node_attributesÿÿÿÿÿÿÿÿÿ"ª
 
z
zÿÿÿÿÿÿÿÿÿ
0
	z_log_var# 
	z_log_varÿÿÿÿÿÿÿÿÿ
*
z_mean 
z_meanÿÿÿÿÿÿÿÿÿË
;__inference_z_layer_call_and_return_conditional_losses_3979b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
;__inference_z_layer_call_and_return_conditional_losses_4001b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
 __inference_z_layer_call_fn_3951~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "ÿÿÿÿÿÿÿÿÿ¢
 __inference_z_layer_call_fn_3957~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_z_log_var_layer_call_and_return_conditional_losses_3945\QR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_z_log_var_layer_call_fn_3935OQR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_z_mean_layer_call_and_return_conditional_losses_3928\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_z_mean_layer_call_fn_3918OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ