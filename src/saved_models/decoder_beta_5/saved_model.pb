??

??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02unknown8??	
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:
*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:
*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:
*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
: *
dtype0
?
adjacency_matrix_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameadjacency_matrix_1/kernel
?
-adjacency_matrix_1/kernel/Read/ReadVariableOpReadVariableOpadjacency_matrix_1/kernel*&
_output_shapes
: *
dtype0
?
adjacency_matrix_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameadjacency_matrix_1/bias

+adjacency_matrix_1/bias/Read/ReadVariableOpReadVariableOpadjacency_matrix_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
?

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
?

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
Z
0
1
 2
!3
(4
)5
06
17
88
99
F10
G11*
Z
0
1
 2
!3
(4
)5
06
17
88
99
F10
G11*
* 
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Sserving_default* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

00
11*

00
11*
* 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 
* 
* 
ic
VARIABLE_VALUEadjacency_matrix_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEadjacency_matrix_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
C
0
1
2
3
4
5
6
7
	8*
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
}
serving_default_z_samplingPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_z_samplingdense_6/kerneldense_6/biasconv2d_transpose/kernelconv2d_transpose/biasdense_4/kerneldense_4/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasdense_5/kerneldense_5/biasadjacency_matrix_1/kerneladjacency_matrix_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_5119
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-adjacency_matrix_1/kernel/Read/ReadVariableOp+adjacency_matrix_1/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *&
f!R
__inference__traced_save_5393
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_4/kerneldense_4/biasconv2d_transpose/kernelconv2d_transpose/biasdense_5/kerneldense_5/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasadjacency_matrix_1/kerneladjacency_matrix_1/bias*
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
GPU2*0J 8? *)
f$R"
 __inference__traced_restore_5439??
?+
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4555

inputs)
dense_6_dense_6_kernel:	?#
dense_6_dense_6_bias:	?B
(conv2d_transpose_conv2d_transpose_kernel:@@4
&conv2d_transpose_conv2d_transpose_bias:@(
dense_4_dense_4_kernel:
"
dense_4_dense_4_bias:
F
,conv2d_transpose_1_conv2d_transpose_1_kernel: @8
*conv2d_transpose_1_conv2d_transpose_1_bias: (
dense_5_dense_5_kernel:
"
dense_5_dense_5_bias:D
*adjacency_matrix_adjacency_matrix_1_kernel: 6
(adjacency_matrix_adjacency_matrix_1_bias:
identity

identity_1??(adjacency_matrix/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_4479?
reshape/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4497?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4513?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4531?
(adjacency_matrix/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*adjacency_matrix_adjacency_matrix_1_kernel(adjacency_matrix_adjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422?
node_attributes/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551{
IdentityIdentity(node_attributes/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????

Identity_1Identity1adjacency_matrix/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^adjacency_matrix/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2T
(adjacency_matrix/StatefulPartitionedCall(adjacency_matrix/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422

inputsS
9conv2d_transpose_readvariableop_adjacency_matrix_1_kernel: <
.biasadd_readvariableop_adjacency_matrix_1_bias:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_adjacency_matrix_1_kernel*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_adjacency_matrix_1_bias*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4993

inputs?
,dense_6_matmul_readvariableop_dense_6_kernel:	?:
+dense_6_biasadd_readvariableop_dense_6_bias:	?b
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel:@@K
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias:@>
,dense_4_matmul_readvariableop_dense_4_kernel:
9
+dense_4_biasadd_readvariableop_dense_4_bias:
f
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel: @O
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias: >
,dense_5_matmul_readvariableop_dense_5_kernel:
9
+dense_5_biasadd_readvariableop_dense_5_bias:d
Jadjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel: M
?adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias:
identity

identity_1??'adjacency_matrix/BiasAdd/ReadVariableOp?0adjacency_matrix/conv2d_transpose/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp,dense_6_matmul_readvariableop_dense_6_kernel*
_output_shapes
:	?*
dtype0z
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
reshape/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_6/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@@*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:
*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:
*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

:
*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
adjacency_matrix/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:n
$adjacency_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&adjacency_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&adjacency_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
adjacency_matrix/strided_sliceStridedSliceadjacency_matrix/Shape:output:0-adjacency_matrix/strided_slice/stack:output:0/adjacency_matrix/strided_slice/stack_1:output:0/adjacency_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
adjacency_matrix/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
adjacency_matrix/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
adjacency_matrix/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
adjacency_matrix/stackPack'adjacency_matrix/strided_slice:output:0!adjacency_matrix/stack/1:output:0!adjacency_matrix/stack/2:output:0!adjacency_matrix/stack/3:output:0*
N*
T0*
_output_shapes
:p
&adjacency_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(adjacency_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(adjacency_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 adjacency_matrix/strided_slice_1StridedSliceadjacency_matrix/stack:output:0/adjacency_matrix/strided_slice_1/stack:output:01adjacency_matrix/strided_slice_1/stack_1:output:01adjacency_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0adjacency_matrix/conv2d_transpose/ReadVariableOpReadVariableOpJadjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel*&
_output_shapes
: *
dtype0?
!adjacency_matrix/conv2d_transposeConv2DBackpropInputadjacency_matrix/stack:output:08adjacency_matrix/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'adjacency_matrix/BiasAdd/ReadVariableOpReadVariableOp?adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias*
_output_shapes
:*
dtype0?
adjacency_matrix/BiasAddBiasAdd*adjacency_matrix/conv2d_transpose:output:0/adjacency_matrix/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
adjacency_matrix/SigmoidSigmoid!adjacency_matrix/BiasAdd:output:0*
T0*/
_output_shapes
:?????????X
node_attributes/ShapeShapedense_5/Sigmoid:y:0*
T0*
_output_shapes
:m
#node_attributes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%node_attributes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%node_attributes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
node_attributes/strided_sliceStridedSlicenode_attributes/Shape:output:0,node_attributes/strided_slice/stack:output:0.node_attributes/strided_slice/stack_1:output:0.node_attributes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
node_attributes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
node_attributes/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
node_attributes/Reshape/shapePack&node_attributes/strided_slice:output:0(node_attributes/Reshape/shape/1:output:0(node_attributes/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
node_attributes/ReshapeReshapedense_5/Sigmoid:y:0&node_attributes/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????s
IdentityIdentity node_attributes/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????u

Identity_1Identityadjacency_matrix/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^adjacency_matrix/BiasAdd/ReadVariableOp1^adjacency_matrix/conv2d_transpose/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2R
'adjacency_matrix/BiasAdd/ReadVariableOp'adjacency_matrix/BiasAdd/ReadVariableOp2d
0adjacency_matrix/conv2d_transpose/ReadVariableOp0adjacency_matrix/conv2d_transpose/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_5333

inputsS
9conv2d_transpose_readvariableop_adjacency_matrix_1_kernel: <
.biasadd_readvariableop_adjacency_matrix_1_bias:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_adjacency_matrix_1_kernel*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_adjacency_matrix_1_bias*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_dense_5_layer_call_fn_5222

inputs 
dense_5_kernel:

dense_5_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_kerneldense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4531o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?$
?
__inference__traced_save_5393
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_adjacency_matrix_1_kernel_read_readvariableop6
2savev2_adjacency_matrix_1_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_adjacency_matrix_1_kernel_read_readvariableop2savev2_adjacency_matrix_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
~: :	?:?:
:
:@@:@:
:: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:$ 

_output_shapes

:
: 

_output_shapes
:
:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:
: 

_output_shapes
::,	(
&
_output_shapes
: @: 


_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?
?
-__inference_reconstruction_layer_call_fn_4869

inputs!
dense_6_kernel:	?
dense_6_bias:	?1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@ 
dense_4_kernel:

dense_4_bias:
3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias:  
dense_5_kernel:

dense_5_bias:3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_biasconv2d_transpose_kernelconv2d_transpose_biasdense_4_kerneldense_4_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasdense_5_kerneldense_5_biasadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reconstruction_layer_call_and_return_conditional_losses_4555s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_reconstruction_layer_call_fn_4572

z_sampling!
dense_6_kernel:	?
dense_6_bias:	?1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@ 
dense_4_kernel:

dense_4_bias:
3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias:  
dense_5_kernel:

dense_5_bias:3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_6_kerneldense_6_biasconv2d_transpose_kernelconv2d_transpose_biasdense_4_kerneldense_4_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasdense_5_kerneldense_5_biasadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reconstruction_layer_call_and_return_conditional_losses_4555s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
?"
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345

inputsS
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel: @<
.biasadd_readvariableop_conv2d_transpose_1_bias: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
]
A__inference_reshape_layer_call_and_return_conditional_losses_4497

inputs
identity;
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

e
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551

inputs
identity;
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_4_layer_call_and_return_conditional_losses_5174

inputs6
$matmul_readvariableop_dense_4_kernel:
1
#biasadd_readvariableop_dense_4_bias:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_reconstruction_layer_call_fn_4800

z_sampling!
dense_6_kernel:	?
dense_6_bias:	?1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@ 
dense_4_kernel:

dense_4_bias:
3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias:  
dense_5_kernel:

dense_5_bias:3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_6_kerneldense_6_biasconv2d_transpose_kernelconv2d_transpose_biasdense_4_kerneldense_4_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasdense_5_kerneldense_5_biasadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reconstruction_layer_call_and_return_conditional_losses_4714s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
?

?
A__inference_dense_6_layer_call_and_return_conditional_losses_4479

inputs7
$matmul_readvariableop_dense_6_kernel:	?2
#biasadd_readvariableop_dense_6_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_6_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5274

inputsS
9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel: @<
.biasadd_readvariableop_conv2d_transpose_1_bias: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
?
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_5119

z_sampling!
dense_6_kernel:	?
dense_6_bias:	?1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@ 
dense_4_kernel:

dense_4_bias:
3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias:  
dense_5_kernel:

dense_5_bias:3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_6_kerneldense_6_biasconv2d_transpose_kernelconv2d_transpose_biasdense_4_kerneldense_4_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasdense_5_kerneldense_5_biasadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_4230w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????u

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
ş
?
__inference__wrapped_model_4230

z_samplingN
;reconstruction_dense_6_matmul_readvariableop_dense_6_kernel:	?I
:reconstruction_dense_6_biasadd_readvariableop_dense_6_bias:	?q
Wreconstruction_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel:@@Z
Lreconstruction_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias:@M
;reconstruction_dense_4_matmul_readvariableop_dense_4_kernel:
H
:reconstruction_dense_4_biasadd_readvariableop_dense_4_bias:
u
[reconstruction_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel: @^
Preconstruction_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias: M
;reconstruction_dense_5_matmul_readvariableop_dense_5_kernel:
H
:reconstruction_dense_5_biasadd_readvariableop_dense_5_bias:s
Yreconstruction_adjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel: \
Nreconstruction_adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias:
identity

identity_1??6reconstruction/adjacency_matrix/BiasAdd/ReadVariableOp??reconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOp?6reconstruction/conv2d_transpose/BiasAdd/ReadVariableOp??reconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOp?8reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOp?Areconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?-reconstruction/dense_4/BiasAdd/ReadVariableOp?,reconstruction/dense_4/MatMul/ReadVariableOp?-reconstruction/dense_5/BiasAdd/ReadVariableOp?,reconstruction/dense_5/MatMul/ReadVariableOp?-reconstruction/dense_6/BiasAdd/ReadVariableOp?,reconstruction/dense_6/MatMul/ReadVariableOp?
,reconstruction/dense_6/MatMul/ReadVariableOpReadVariableOp;reconstruction_dense_6_matmul_readvariableop_dense_6_kernel*
_output_shapes
:	?*
dtype0?
reconstruction/dense_6/MatMulMatMul
z_sampling4reconstruction/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-reconstruction/dense_6/BiasAdd/ReadVariableOpReadVariableOp:reconstruction_dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype0?
reconstruction/dense_6/BiasAddBiasAdd'reconstruction/dense_6/MatMul:product:05reconstruction/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
reconstruction/dense_6/ReluRelu'reconstruction/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????u
reconstruction/reshape/ShapeShape)reconstruction/dense_6/Relu:activations:0*
T0*
_output_shapes
:t
*reconstruction/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,reconstruction/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,reconstruction/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$reconstruction/reshape/strided_sliceStridedSlice%reconstruction/reshape/Shape:output:03reconstruction/reshape/strided_slice/stack:output:05reconstruction/reshape/strided_slice/stack_1:output:05reconstruction/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&reconstruction/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&reconstruction/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :h
&reconstruction/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
$reconstruction/reshape/Reshape/shapePack-reconstruction/reshape/strided_slice:output:0/reconstruction/reshape/Reshape/shape/1:output:0/reconstruction/reshape/Reshape/shape/2:output:0/reconstruction/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reconstruction/reshape/ReshapeReshape)reconstruction/dense_6/Relu:activations:0-reconstruction/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@|
%reconstruction/conv2d_transpose/ShapeShape'reconstruction/reshape/Reshape:output:0*
T0*
_output_shapes
:}
3reconstruction/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5reconstruction/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5reconstruction/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-reconstruction/conv2d_transpose/strided_sliceStridedSlice.reconstruction/conv2d_transpose/Shape:output:0<reconstruction/conv2d_transpose/strided_slice/stack:output:0>reconstruction/conv2d_transpose/strided_slice/stack_1:output:0>reconstruction/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'reconstruction/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :i
'reconstruction/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :i
'reconstruction/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
%reconstruction/conv2d_transpose/stackPack6reconstruction/conv2d_transpose/strided_slice:output:00reconstruction/conv2d_transpose/stack/1:output:00reconstruction/conv2d_transpose/stack/2:output:00reconstruction/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:
5reconstruction/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7reconstruction/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7reconstruction/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/reconstruction/conv2d_transpose/strided_slice_1StridedSlice.reconstruction/conv2d_transpose/stack:output:0>reconstruction/conv2d_transpose/strided_slice_1/stack:output:0@reconstruction/conv2d_transpose/strided_slice_1/stack_1:output:0@reconstruction/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?reconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpWreconstruction_conv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@@*
dtype0?
0reconstruction/conv2d_transpose/conv2d_transposeConv2DBackpropInput.reconstruction/conv2d_transpose/stack:output:0Greconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0'reconstruction/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
6reconstruction/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpLreconstruction_conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype0?
'reconstruction/conv2d_transpose/BiasAddBiasAdd9reconstruction/conv2d_transpose/conv2d_transpose:output:0>reconstruction/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
$reconstruction/conv2d_transpose/ReluRelu0reconstruction/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
,reconstruction/dense_4/MatMul/ReadVariableOpReadVariableOp;reconstruction_dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:
*
dtype0?
reconstruction/dense_4/MatMulMatMul
z_sampling4reconstruction/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
-reconstruction/dense_4/BiasAdd/ReadVariableOpReadVariableOp:reconstruction_dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:
*
dtype0?
reconstruction/dense_4/BiasAddBiasAdd'reconstruction/dense_4/MatMul:product:05reconstruction/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
~
reconstruction/dense_4/ReluRelu'reconstruction/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
'reconstruction/conv2d_transpose_1/ShapeShape2reconstruction/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:
5reconstruction/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7reconstruction/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7reconstruction/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/reconstruction/conv2d_transpose_1/strided_sliceStridedSlice0reconstruction/conv2d_transpose_1/Shape:output:0>reconstruction/conv2d_transpose_1/strided_slice/stack:output:0@reconstruction/conv2d_transpose_1/strided_slice/stack_1:output:0@reconstruction/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)reconstruction/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)reconstruction/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)reconstruction/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
'reconstruction/conv2d_transpose_1/stackPack8reconstruction/conv2d_transpose_1/strided_slice:output:02reconstruction/conv2d_transpose_1/stack/1:output:02reconstruction/conv2d_transpose_1/stack/2:output:02reconstruction/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:?
7reconstruction/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9reconstruction/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9reconstruction/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1reconstruction/conv2d_transpose_1/strided_slice_1StridedSlice0reconstruction/conv2d_transpose_1/stack:output:0@reconstruction/conv2d_transpose_1/strided_slice_1/stack:output:0Breconstruction/conv2d_transpose_1/strided_slice_1/stack_1:output:0Breconstruction/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Areconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp[reconstruction_conv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype0?
2reconstruction/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput0reconstruction/conv2d_transpose_1/stack:output:0Ireconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:02reconstruction/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
8reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpPreconstruction_conv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype0?
)reconstruction/conv2d_transpose_1/BiasAddBiasAdd;reconstruction/conv2d_transpose_1/conv2d_transpose:output:0@reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
&reconstruction/conv2d_transpose_1/ReluRelu2reconstruction/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
,reconstruction/dense_5/MatMul/ReadVariableOpReadVariableOp;reconstruction_dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

:
*
dtype0?
reconstruction/dense_5/MatMulMatMul)reconstruction/dense_4/Relu:activations:04reconstruction/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-reconstruction/dense_5/BiasAdd/ReadVariableOpReadVariableOp:reconstruction_dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0?
reconstruction/dense_5/BiasAddBiasAdd'reconstruction/dense_5/MatMul:product:05reconstruction/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
reconstruction/dense_5/SigmoidSigmoid'reconstruction/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%reconstruction/adjacency_matrix/ShapeShape4reconstruction/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:}
3reconstruction/adjacency_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5reconstruction/adjacency_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5reconstruction/adjacency_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-reconstruction/adjacency_matrix/strided_sliceStridedSlice.reconstruction/adjacency_matrix/Shape:output:0<reconstruction/adjacency_matrix/strided_slice/stack:output:0>reconstruction/adjacency_matrix/strided_slice/stack_1:output:0>reconstruction/adjacency_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'reconstruction/adjacency_matrix/stack/1Const*
_output_shapes
: *
dtype0*
value	B :i
'reconstruction/adjacency_matrix/stack/2Const*
_output_shapes
: *
dtype0*
value	B :i
'reconstruction/adjacency_matrix/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
%reconstruction/adjacency_matrix/stackPack6reconstruction/adjacency_matrix/strided_slice:output:00reconstruction/adjacency_matrix/stack/1:output:00reconstruction/adjacency_matrix/stack/2:output:00reconstruction/adjacency_matrix/stack/3:output:0*
N*
T0*
_output_shapes
:
5reconstruction/adjacency_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7reconstruction/adjacency_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7reconstruction/adjacency_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/reconstruction/adjacency_matrix/strided_slice_1StridedSlice.reconstruction/adjacency_matrix/stack:output:0>reconstruction/adjacency_matrix/strided_slice_1/stack:output:0@reconstruction/adjacency_matrix/strided_slice_1/stack_1:output:0@reconstruction/adjacency_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
?reconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOpReadVariableOpYreconstruction_adjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel*&
_output_shapes
: *
dtype0?
0reconstruction/adjacency_matrix/conv2d_transposeConv2DBackpropInput.reconstruction/adjacency_matrix/stack:output:0Greconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOp:value:04reconstruction/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
6reconstruction/adjacency_matrix/BiasAdd/ReadVariableOpReadVariableOpNreconstruction_adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias*
_output_shapes
:*
dtype0?
'reconstruction/adjacency_matrix/BiasAddBiasAdd9reconstruction/adjacency_matrix/conv2d_transpose:output:0>reconstruction/adjacency_matrix/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
'reconstruction/adjacency_matrix/SigmoidSigmoid0reconstruction/adjacency_matrix/BiasAdd:output:0*
T0*/
_output_shapes
:?????????v
$reconstruction/node_attributes/ShapeShape"reconstruction/dense_5/Sigmoid:y:0*
T0*
_output_shapes
:|
2reconstruction/node_attributes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4reconstruction/node_attributes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4reconstruction/node_attributes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,reconstruction/node_attributes/strided_sliceStridedSlice-reconstruction/node_attributes/Shape:output:0;reconstruction/node_attributes/strided_slice/stack:output:0=reconstruction/node_attributes/strided_slice/stack_1:output:0=reconstruction/node_attributes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.reconstruction/node_attributes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :p
.reconstruction/node_attributes/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
,reconstruction/node_attributes/Reshape/shapePack5reconstruction/node_attributes/strided_slice:output:07reconstruction/node_attributes/Reshape/shape/1:output:07reconstruction/node_attributes/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
&reconstruction/node_attributes/ReshapeReshape"reconstruction/dense_5/Sigmoid:y:05reconstruction/node_attributes/Reshape/shape:output:0*
T0*+
_output_shapes
:??????????
IdentityIdentity+reconstruction/adjacency_matrix/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity/reconstruction/node_attributes/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp7^reconstruction/adjacency_matrix/BiasAdd/ReadVariableOp@^reconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOp7^reconstruction/conv2d_transpose/BiasAdd/ReadVariableOp@^reconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOp9^reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOpB^reconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOp.^reconstruction/dense_4/BiasAdd/ReadVariableOp-^reconstruction/dense_4/MatMul/ReadVariableOp.^reconstruction/dense_5/BiasAdd/ReadVariableOp-^reconstruction/dense_5/MatMul/ReadVariableOp.^reconstruction/dense_6/BiasAdd/ReadVariableOp-^reconstruction/dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2p
6reconstruction/adjacency_matrix/BiasAdd/ReadVariableOp6reconstruction/adjacency_matrix/BiasAdd/ReadVariableOp2?
?reconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOp?reconstruction/adjacency_matrix/conv2d_transpose/ReadVariableOp2p
6reconstruction/conv2d_transpose/BiasAdd/ReadVariableOp6reconstruction/conv2d_transpose/BiasAdd/ReadVariableOp2?
?reconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOp?reconstruction/conv2d_transpose/conv2d_transpose/ReadVariableOp2t
8reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOp8reconstruction/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Areconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOpAreconstruction/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^
-reconstruction/dense_4/BiasAdd/ReadVariableOp-reconstruction/dense_4/BiasAdd/ReadVariableOp2\
,reconstruction/dense_4/MatMul/ReadVariableOp,reconstruction/dense_4/MatMul/ReadVariableOp2^
-reconstruction/dense_5/BiasAdd/ReadVariableOp-reconstruction/dense_5/BiasAdd/ReadVariableOp2\
,reconstruction/dense_5/MatMul/ReadVariableOp,reconstruction/dense_5/MatMul/ReadVariableOp2^
-reconstruction/dense_6/BiasAdd/ReadVariableOp-reconstruction/dense_6/BiasAdd/ReadVariableOp2\
,reconstruction/dense_6/MatMul/ReadVariableOp,reconstruction/dense_6/MatMul/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
?
]
A__inference_reshape_layer_call_and_return_conditional_losses_5156

inputs
identity;
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5215

inputsQ
7conv2d_transpose_readvariableop_conv2d_transpose_kernel:@@:
,biasadd_readvariableop_conv2d_transpose_bias:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides

BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
B
&__inference_reshape_layer_call_fn_5142

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4497h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_adjacency_matrix_layer_call_fn_5299

inputs3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?+
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4825

z_sampling)
dense_6_dense_6_kernel:	?#
dense_6_dense_6_bias:	?B
(conv2d_transpose_conv2d_transpose_kernel:@@4
&conv2d_transpose_conv2d_transpose_bias:@(
dense_4_dense_4_kernel:
"
dense_4_dense_4_bias:
F
,conv2d_transpose_1_conv2d_transpose_1_kernel: @8
*conv2d_transpose_1_conv2d_transpose_1_bias: (
dense_5_dense_5_kernel:
"
dense_5_dense_5_bias:D
*adjacency_matrix_adjacency_matrix_1_kernel: 6
(adjacency_matrix_adjacency_matrix_1_bias:
identity

identity_1??(adjacency_matrix/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_4479?
reshape/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4497?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268?
dense_4/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4513?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4531?
(adjacency_matrix/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*adjacency_matrix_adjacency_matrix_1_kernel(adjacency_matrix_adjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422?
node_attributes/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551{
IdentityIdentity(node_attributes/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????

Identity_1Identity1adjacency_matrix/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^adjacency_matrix/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2T
(adjacency_matrix/StatefulPartitionedCall(adjacency_matrix/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
?
?
-__inference_reconstruction_layer_call_fn_4888

inputs!
dense_6_kernel:	?
dense_6_bias:	?1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@ 
dense_4_kernel:

dense_4_bias:
3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias:  
dense_5_kernel:

dense_5_bias:3
adjacency_matrix_1_kernel: %
adjacency_matrix_1_bias:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_biasconv2d_transpose_kernelconv2d_transpose_biasdense_4_kerneldense_4_biasconv2d_transpose_1_kernelconv2d_transpose_1_biasdense_5_kerneldense_5_biasadjacency_matrix_1_kerneladjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *F
_output_shapes4
2:?????????:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_reconstruction_layer_call_and_return_conditional_losses_4714s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_4_layer_call_and_return_conditional_losses_4513

inputs6
$matmul_readvariableop_dense_4_kernel:
1
#biasadd_readvariableop_dense_4_bias:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_4_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_4_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_conv2d_transpose_layer_call_fn_5181

inputs1
conv2d_transpose_kernel:@@#
conv2d_transpose_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_kernelconv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
&__inference_dense_6_layer_call_fn_5126

inputs!
dense_6_kernel:	?
dense_6_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_kerneldense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_4479p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_node_attributes_layer_call_fn_5279

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_5_layer_call_and_return_conditional_losses_5233

inputs6
$matmul_readvariableop_dense_5_kernel:
1
#biasadd_readvariableop_dense_5_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
A__inference_dense_6_layer_call_and_return_conditional_losses_5137

inputs7
$matmul_readvariableop_dense_6_kernel:	?2
#biasadd_readvariableop_dense_6_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_6_kernel*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_dense_5_layer_call_and_return_conditional_losses_4531

inputs6
$matmul_readvariableop_dense_5_kernel:
1
#biasadd_readvariableop_dense_5_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_5_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?+
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4850

z_sampling)
dense_6_dense_6_kernel:	?#
dense_6_dense_6_bias:	?B
(conv2d_transpose_conv2d_transpose_kernel:@@4
&conv2d_transpose_conv2d_transpose_bias:@(
dense_4_dense_4_kernel:
"
dense_4_dense_4_bias:
F
,conv2d_transpose_1_conv2d_transpose_1_kernel: @8
*conv2d_transpose_1_conv2d_transpose_1_bias: (
dense_5_dense_5_kernel:
"
dense_5_dense_5_bias:D
*adjacency_matrix_adjacency_matrix_1_kernel: 6
(adjacency_matrix_adjacency_matrix_1_bias:
identity

identity_1??(adjacency_matrix/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_4479?
reshape/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4497?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268?
dense_4/StatefulPartitionedCallStatefulPartitionedCall
z_samplingdense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4513?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4531?
(adjacency_matrix/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*adjacency_matrix_adjacency_matrix_1_kernel(adjacency_matrix_adjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422?
node_attributes/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551{
IdentityIdentity(node_attributes/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????

Identity_1Identity1adjacency_matrix/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^adjacency_matrix/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2T
(adjacency_matrix/StatefulPartitionedCall(adjacency_matrix/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
z_sampling
?

e
I__inference_node_attributes_layer_call_and_return_conditional_losses_5292

inputs
identity;
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4714

inputs)
dense_6_dense_6_kernel:	?#
dense_6_dense_6_bias:	?B
(conv2d_transpose_conv2d_transpose_kernel:@@4
&conv2d_transpose_conv2d_transpose_bias:@(
dense_4_dense_4_kernel:
"
dense_4_dense_4_bias:
F
,conv2d_transpose_1_conv2d_transpose_1_kernel: @8
*conv2d_transpose_1_conv2d_transpose_1_bias: (
dense_5_dense_5_kernel:
"
dense_5_dense_5_bias:D
*adjacency_matrix_adjacency_matrix_1_kernel: 6
(adjacency_matrix_adjacency_matrix_1_bias:
identity

identity_1??(adjacency_matrix/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_dense_6_kerneldense_6_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_4479?
reshape/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4497?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0(conv2d_transpose_conv2d_transpose_kernel&conv2d_transpose_conv2d_transpose_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_dense_4_kerneldense_4_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4513?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0,conv2d_transpose_1_conv2d_transpose_1_kernel*conv2d_transpose_1_conv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_dense_5_kerneldense_5_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_4531?
(adjacency_matrix/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0*adjacency_matrix_adjacency_matrix_1_kernel(adjacency_matrix_adjacency_matrix_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_4422?
node_attributes/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_node_attributes_layer_call_and_return_conditional_losses_4551{
IdentityIdentity(node_attributes/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????

Identity_1Identity1adjacency_matrix/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^adjacency_matrix/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2T
(adjacency_matrix/StatefulPartitionedCall(adjacency_matrix/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_dense_4_layer_call_fn_5163

inputs 
dense_4_kernel:

dense_4_bias:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_kerneldense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_4513o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_1_layer_call_fn_5240

inputs3
conv2d_transpose_1_kernel: @%
conv2d_transpose_1_bias: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_1_kernelconv2d_transpose_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4345?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?!
?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4268

inputsQ
7conv2d_transpose_readvariableop_conv2d_transpose_kernel:@@:
,biasadd_readvariableop_conv2d_transpose_bias:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp7conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@@*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides

BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
H__inference_reconstruction_layer_call_and_return_conditional_losses_5098

inputs?
,dense_6_matmul_readvariableop_dense_6_kernel:	?:
+dense_6_biasadd_readvariableop_dense_6_bias:	?b
Hconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel:@@K
=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias:@>
,dense_4_matmul_readvariableop_dense_4_kernel:
9
+dense_4_biasadd_readvariableop_dense_4_bias:
f
Lconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel: @O
Aconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias: >
,dense_5_matmul_readvariableop_dense_5_kernel:
9
+dense_5_biasadd_readvariableop_dense_5_bias:d
Jadjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel: M
?adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias:
identity

identity_1??'adjacency_matrix/BiasAdd/ReadVariableOp?0adjacency_matrix/conv2d_transpose/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp,dense_6_matmul_readvariableop_dense_6_kernel*
_output_shapes
:	?*
dtype0z
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp+dense_6_biasadd_readvariableop_dense_6_bias*
_output_shapes	
:?*
dtype0?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????W
reshape/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_6/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpHconv2d_transpose_conv2d_transpose_readvariableop_conv2d_transpose_kernel*&
_output_shapes
:@@*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=conv2d_transpose_biasadd_readvariableop_conv2d_transpose_bias*
_output_shapes
:@*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
dense_4/MatMul/ReadVariableOpReadVariableOp,dense_4_matmul_readvariableop_dense_4_kernel*
_output_shapes

:
*
dtype0y
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_4/BiasAdd/ReadVariableOpReadVariableOp+dense_4_biasadd_readvariableop_dense_4_bias*
_output_shapes
:
*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpLconv2d_transpose_1_conv2d_transpose_readvariableop_conv2d_transpose_1_kernel*&
_output_shapes
: @*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpAconv2d_transpose_1_biasadd_readvariableop_conv2d_transpose_1_bias*
_output_shapes
: *
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
dense_5/MatMul/ReadVariableOpReadVariableOp,dense_5_matmul_readvariableop_dense_5_kernel*
_output_shapes

:
*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp+dense_5_biasadd_readvariableop_dense_5_bias*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
adjacency_matrix/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:n
$adjacency_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&adjacency_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&adjacency_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
adjacency_matrix/strided_sliceStridedSliceadjacency_matrix/Shape:output:0-adjacency_matrix/strided_slice/stack:output:0/adjacency_matrix/strided_slice/stack_1:output:0/adjacency_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
adjacency_matrix/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
adjacency_matrix/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
adjacency_matrix/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
adjacency_matrix/stackPack'adjacency_matrix/strided_slice:output:0!adjacency_matrix/stack/1:output:0!adjacency_matrix/stack/2:output:0!adjacency_matrix/stack/3:output:0*
N*
T0*
_output_shapes
:p
&adjacency_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(adjacency_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(adjacency_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 adjacency_matrix/strided_slice_1StridedSliceadjacency_matrix/stack:output:0/adjacency_matrix/strided_slice_1/stack:output:01adjacency_matrix/strided_slice_1/stack_1:output:01adjacency_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0adjacency_matrix/conv2d_transpose/ReadVariableOpReadVariableOpJadjacency_matrix_conv2d_transpose_readvariableop_adjacency_matrix_1_kernel*&
_output_shapes
: *
dtype0?
!adjacency_matrix/conv2d_transposeConv2DBackpropInputadjacency_matrix/stack:output:08adjacency_matrix/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'adjacency_matrix/BiasAdd/ReadVariableOpReadVariableOp?adjacency_matrix_biasadd_readvariableop_adjacency_matrix_1_bias*
_output_shapes
:*
dtype0?
adjacency_matrix/BiasAddBiasAdd*adjacency_matrix/conv2d_transpose:output:0/adjacency_matrix/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
adjacency_matrix/SigmoidSigmoid!adjacency_matrix/BiasAdd:output:0*
T0*/
_output_shapes
:?????????X
node_attributes/ShapeShapedense_5/Sigmoid:y:0*
T0*
_output_shapes
:m
#node_attributes/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%node_attributes/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%node_attributes/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
node_attributes/strided_sliceStridedSlicenode_attributes/Shape:output:0,node_attributes/strided_slice/stack:output:0.node_attributes/strided_slice/stack_1:output:0.node_attributes/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
node_attributes/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
node_attributes/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
node_attributes/Reshape/shapePack&node_attributes/strided_slice:output:0(node_attributes/Reshape/shape/1:output:0(node_attributes/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
node_attributes/ReshapeReshapedense_5/Sigmoid:y:0&node_attributes/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????s
IdentityIdentity node_attributes/Reshape:output:0^NoOp*
T0*+
_output_shapes
:?????????u

Identity_1Identityadjacency_matrix/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^adjacency_matrix/BiasAdd/ReadVariableOp1^adjacency_matrix/conv2d_transpose/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????: : : : : : : : : : : : 2R
'adjacency_matrix/BiasAdd/ReadVariableOp'adjacency_matrix/BiasAdd/ReadVariableOp2d
0adjacency_matrix/conv2d_transpose/ReadVariableOp0adjacency_matrix/conv2d_transpose/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
 __inference__traced_restore_5439
file_prefix2
assignvariableop_dense_6_kernel:	?.
assignvariableop_1_dense_6_bias:	?3
!assignvariableop_2_dense_4_kernel:
-
assignvariableop_3_dense_4_bias:
D
*assignvariableop_4_conv2d_transpose_kernel:@@6
(assignvariableop_5_conv2d_transpose_bias:@3
!assignvariableop_6_dense_5_kernel:
-
assignvariableop_7_dense_5_bias:F
,assignvariableop_8_conv2d_transpose_1_kernel: @8
*assignvariableop_9_conv2d_transpose_1_bias: G
-assignvariableop_10_adjacency_matrix_1_kernel: 9
+assignvariableop_11_adjacency_matrix_1_bias:
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_conv2d_transpose_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_conv2d_transpose_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2d_transpose_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_transpose_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_adjacency_matrix_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adjacency_matrix_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: ?
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
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A

z_sampling3
serving_default_z_sampling:0?????????L
adjacency_matrix8
StatefulPartitionedCall:0?????????G
node_attributes4
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
v
0
1
 2
!3
(4
)5
06
17
88
99
F10
G11"
trackable_list_wrapper
v
0
1
 2
!3
(4
)5
06
17
88
99
F10
G11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_reconstruction_layer_call_fn_4572
-__inference_reconstruction_layer_call_fn_4869
-__inference_reconstruction_layer_call_fn_4888
-__inference_reconstruction_layer_call_fn_4800?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4993
H__inference_reconstruction_layer_call_and_return_conditional_losses_5098
H__inference_reconstruction_layer_call_and_return_conditional_losses_4825
H__inference_reconstruction_layer_call_and_return_conditional_losses_4850?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_4230
z_sampling"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
Sserving_default"
signature_map
!:	?2dense_6/kernel
:?2dense_6/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_6_layer_call_fn_5126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_6_layer_call_and_return_conditional_losses_5137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_reshape_layer_call_fn_5142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_reshape_layer_call_and_return_conditional_losses_5156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :
2dense_4/kernel
:
2dense_4/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_4_layer_call_fn_5163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_4_layer_call_and_return_conditional_losses_5174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
1:/@@2conv2d_transpose/kernel
#:!@2conv2d_transpose/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_conv2d_transpose_layer_call_fn_5181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5215?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :
2dense_5/kernel
:2dense_5/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_5_layer_call_fn_5222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_5_layer_call_and_return_conditional_losses_5233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3:1 @2conv2d_transpose_1/kernel
%:# 2conv2d_transpose_1/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_conv2d_transpose_1_layer_call_fn_5240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_node_attributes_layer_call_fn_5279?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_node_attributes_layer_call_and_return_conditional_losses_5292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3:1 2adjacency_matrix_1/kernel
%:#2adjacency_matrix_1/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_adjacency_matrix_layer_call_fn_5299?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_5333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
"__inference_signature_wrapper_5119
z_sampling"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
trackable_dict_wrapper?
__inference__wrapped_model_4230?() !8901FG3?0
)?&
$?!

z_sampling?????????
? "???
F
adjacency_matrix2?/
adjacency_matrix?????????
@
node_attributes-?*
node_attributes??????????
J__inference_adjacency_matrix_layer_call_and_return_conditional_losses_5333?FGI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
/__inference_adjacency_matrix_layer_call_fn_5299?FGI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_5274?89I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
1__inference_conv2d_transpose_1_layer_call_fn_5240?89I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_5215?()I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
/__inference_conv2d_transpose_layer_call_fn_5181?()I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
A__inference_dense_4_layer_call_and_return_conditional_losses_5174\ !/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? y
&__inference_dense_4_layer_call_fn_5163O !/?,
%?"
 ?
inputs?????????
? "??????????
?
A__inference_dense_5_layer_call_and_return_conditional_losses_5233\01/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? y
&__inference_dense_5_layer_call_fn_5222O01/?,
%?"
 ?
inputs?????????

? "???????????
A__inference_dense_6_layer_call_and_return_conditional_losses_5137]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? z
&__inference_dense_6_layer_call_fn_5126P/?,
%?"
 ?
inputs?????????
? "????????????
I__inference_node_attributes_layer_call_and_return_conditional_losses_5292\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
.__inference_node_attributes_layer_call_fn_5279O/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_reconstruction_layer_call_and_return_conditional_losses_4825?() !8901FG;?8
1?.
$?!

z_sampling?????????
p 

 
? "W?T
M?J
!?
0/0?????????
%?"
0/1?????????
? ?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4850?() !8901FG;?8
1?.
$?!

z_sampling?????????
p

 
? "W?T
M?J
!?
0/0?????????
%?"
0/1?????????
? ?
H__inference_reconstruction_layer_call_and_return_conditional_losses_4993?() !8901FG7?4
-?*
 ?
inputs?????????
p 

 
? "W?T
M?J
!?
0/0?????????
%?"
0/1?????????
? ?
H__inference_reconstruction_layer_call_and_return_conditional_losses_5098?() !8901FG7?4
-?*
 ?
inputs?????????
p

 
? "W?T
M?J
!?
0/0?????????
%?"
0/1?????????
? ?
-__inference_reconstruction_layer_call_fn_4572?() !8901FG;?8
1?.
$?!

z_sampling?????????
p 

 
? "I?F
?
0?????????
#? 
1??????????
-__inference_reconstruction_layer_call_fn_4800?() !8901FG;?8
1?.
$?!

z_sampling?????????
p

 
? "I?F
?
0?????????
#? 
1??????????
-__inference_reconstruction_layer_call_fn_4869?() !8901FG7?4
-?*
 ?
inputs?????????
p 

 
? "I?F
?
0?????????
#? 
1??????????
-__inference_reconstruction_layer_call_fn_4888?() !8901FG7?4
-?*
 ?
inputs?????????
p

 
? "I?F
?
0?????????
#? 
1??????????
A__inference_reshape_layer_call_and_return_conditional_losses_5156a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????@
? ~
&__inference_reshape_layer_call_fn_5142T0?-
&?#
!?
inputs??????????
? " ??????????@?
"__inference_signature_wrapper_5119?() !8901FGA?>
? 
7?4
2

z_sampling$?!

z_sampling?????????"???
F
adjacency_matrix2?/
adjacency_matrix?????????
@
node_attributes-?*
node_attributes?????????