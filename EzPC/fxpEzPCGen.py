from __future__ import annotations

import re

# code to print
initTape = []
fwdInputs = [('int64_al[2]', 'lossAndAcc'), ('int64_pl[5][SigmoidNumSeg]', 'SigmoidSegData'), ('int64_pl[5][TanhNumSeg]', 'TanhSegData')]
fwdTape = []
zeroGradTape = []
bwdInputs = [('int64_pl', 'AdamAlphaT'), ('int64_pl[5][RSqrtNumSeg]', 'RSqrtSegData')]
bwdTape = []
paramUpdateTape = []

# write code to define and initialize gradient moments to zero
def writeInit(filename: str) -> None:
    outfile = open(filename, 'w')
    for line in initTape:
        outfile.write(line)
    # iteration index - used by ADAM
    outfile.write('int32_pl idxIteration = 0;\n')
    outfile.close()

# write a function that does only forward pass, and a call to that function
def writeTest(funcFilename: str, callFilename: str) -> None:
    outfile = open(funcFilename, 'w')
    outfile.write('def void TestIteration(\n')
    outfile.write(',\n'.join([paramSpec[0] + ' ' + paramSpec[1] for paramSpec in fwdInputs]))
    outfile.write(') {\n')
    for line in fwdTape:
        outfile.write(line)
    outfile.write('}\n')
    outfile.close()
    insertTempFreesAfterLastUses(funcFilename)
    outfile = open(callFilename, 'w')
    outfile.write('TestIteration(')
    outfile.write(', '.join([paramSpec[1] for paramSpec in fwdInputs]))
    outfile.write(');\n')
    outfile.close()

# write a function that does both forward and backward passes, and a call to that function
def writeTrain(funcFilename: str, callFilename: str) -> None:
    outfile = open(funcFilename, 'w')
    outfile.write('def void TrainIteration(\n')
    outfile.write(',\n'.join([paramSpec[0] + ' ' + paramSpec[1] for paramSpec in fwdInputs + bwdInputs]))
    outfile.write(') {\n')
    for line in fwdTape:
        outfile.write(line)
    for line in zeroGradTape:
        outfile.write(line)
    for line in reversed(bwdTape):
        outfile.write(line)
    for line in paramUpdateTape:
        outfile.write(line)
    outfile.write('}\n')
    outfile.close()
    insertTempFreesAfterLastUses(funcFilename)
    outfile = open(callFilename, 'w')
    # write code to update ADAM settings with each iteration, as required by it
    outfile.write('idxIteration = idxIteration + 1;\n'\
                  'int64_pl AdamAlphaT = CalcAlphaT(idxIteration);\n'\
                  'TrainIteration(')
    outfile.write(', '.join([paramSpec[1] for paramSpec in fwdInputs]))
    outfile.write(', ')
    outfile.write(', '.join([paramSpec[1] for paramSpec in bwdInputs]))
    outfile.write(');\n')
    outfile.close()

# free temporary variables after their last use
def insertTempFreesAfterLastUses(filename: str) -> None:
    # detects temp variables and their gradients
    tempDetector = re.compile(r'tnsr[0-9]+(?:Grad)?')
    tempDeclDetector = re.compile(r'int64_al\[\w+\](\[\w+\])? (tnsr[0-9]+(?:Grad)?)')
    # get all the lines
    outfile = open(filename, 'r')
    lines = outfile.readlines()
    outfile.close()
    # get all mentions of temp vars and map them to their dimensionality
    dims = {}
    mentions = []
    for line in lines:
        mentions.append(tempDetector.findall(line))
        decls = tempDeclDetector.findall(line)
        for decl in decls:
            # if only 1 [] in declaration, it is 1D
            dims[decl[1]] = 1 if decl[0] == '' else 2
    # free calls to insert for temp vars and their grads
    insertions = []
    for tempIdx in range(tempCounter):
        tempName = 'tnsr' + str(tempIdx)
        for lineIdx in range(len(lines) - 1, -1, -1):
            if tempName in mentions[lineIdx]:
                insertions.append((lineIdx, tempName))
                break
        tempGradName = tempName + 'Grad'
        if tempGradName in dims:
            for lineIdx in range(len(lines) - 1, -1, -1):
                if tempGradName in mentions[lineIdx]:
                    insertions.append((lineIdx, tempGradName))
                    break
    # sort insertions backwards by line # after which insertion must be done
    insertions.sort()
    insertions.reverse()
    # insert free calls
    for insertion in insertions:
        beforeLine, tempName = insertion
        insertionTemplate = 'Free{ndims}D({temp});\n'
        insertionCode = insertionTemplate.format(ndims = dims[tempName],
                                                 temp = tempName)
        lines.insert(beforeLine + 1, insertionCode)
    # write back file
    outfile = open(filename, 'w')
    for line in lines:
        outfile.write(line)
    outfile.close()

def anyInputRequiresGrad(*args: Tensor) -> bool:
    return any([arg.requiresGrad for arg in args])

def shapeToArrayDefString(shape: tuple) -> str:
    return ''.join(['[' + dim + ']' for dim in shape])

def shapeToFuncCallString(shape: tuple) -> str:
    return ''.join([dim + ', ' for dim in shape])

tempCounter = 0 # automatic naming of tensors
def getTemp() -> str:
    global tempCounter
    tempName = 'tnsr' + str(tempCounter)
    tempCounter += 1
    return tempName

# modeled after pytorch's Tensor datatype
class Tensor:
    def __init__(self, shape: tuple, requiresGrad: bool, isParam: bool = False, name: str = None) -> None:
        # shape for appropriate code gen
        self.shape = shape
        # does gradient of this need to be calculated?
        self.requiresGrad = requiresGrad
        # some common strings
        shapeArrayDefString = shapeToArrayDefString(shape)
        shapeFuncCallString = shapeToFuncCallString(shape)
        # name of this tensor
        if name is not None:
            self.name = name
            # assume has name -> input
            inputParam = ('int64_al' + shapeArrayDefString, self.name)
            fwdInputs.append(inputParam)
        else:
            self.name = getTemp()
        if requiresGrad:
            # if needs gradient, we need to define and zero it prior to starting backward pass
            zeroGradTemplate = 'int64_al{shape} {param}Grad;\n'\
                               'SetToZero{ndims}D({idims}{param}Grad);\n'
            zeroGradCode = zeroGradTemplate.format(shape = shapeArrayDefString,
                                                   param = self.name,
                                                   ndims = len(shape),
                                                   idims = shapeFuncCallString)
            zeroGradTape.append(zeroGradCode)
        if isParam:
            # code to get param from server
            getParamTemplate = 'input(SERVER, {param}, int64_al{shape});\n'
            getParamCode = getParamTemplate.format(param = self.name,
                                                   shape = shapeArrayDefString)
            initTape.append(getParamCode)
            # code to define and zero param grad moments
            zeroGradMomentTemplate = 'int64_al{shape} {param}GradMu;\n'\
                                     'int64_al{shape} {param}GradNu;\n'\
                                     'SetToZero{ndims}D({idims}{param}GradMu);\n'\
                                     'SetToZero{ndims}D({idims}{param}GradNu);\n'
            zeroGradMomentCode = zeroGradMomentTemplate.format(shape = shapeArrayDefString,
                                                               param = self.name,
                                                               ndims = len(shape),
                                                               idims = shapeFuncCallString)
            initTape.append(zeroGradMomentCode)
            # param grad moments are inputs
            inputParam = ('int64_al' + shapeArrayDefString, self.name + 'GradMu')
            bwdInputs.append(inputParam)
            inputParam = ('int64_al' + shapeArrayDefString, self.name + 'GradNu')
            bwdInputs.append(inputParam)
            # code to update param after backward pass
            updateTemplate = 'AdamUpdate{ndims}D({idims}{param}, {param}Grad, {param}GradMu, {param}GradNu, AdamAlphaT, RSqrtSegData);\n'
            updateCode = updateTemplate.format(ndims = len(shape), idims = shapeFuncCallString, param = self.name)
            paramUpdateTape.append(updateCode)

    def __add__(self, othr: Tensor) -> Tensor:
        # auto-select between standard and bias addition
        if len(self.shape) == 2 and len(othr.shape) == 1:
            return addBias(self, othr)
        elif len(self.shape) == len(othr.shape):
            return add(self, othr)
        else:
            raise NotImplementedError

    def __sub__(self, othr: Tensor) -> Tensor:
        return sub(self, othr)

    def __mul__(self, othr: Tensor) -> Tensor:
        return mul(self, othr)

# modeled after pytorch's autograd.Function
class Func:
    @staticmethod
    def apply(*args: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

# add 2 same shaped tensors
class AddFunc(Func):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        ndims = len(a.shape)
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a, b))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'Add{ndims}D({idims}{a}, {b}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, ndims = ndims,
                                     idims = idims, a = a.name, b = b.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        bwdTemplate = 'Add{ndims}D({idims}{result}Grad, {inp}Grad, {inp}Grad);\n'
        if a.requiresGrad:
            bwdCode += bwdTemplate.format(ndims = ndims, idims = idims,
                                          result = result.name, inp = a.name)
        if b.requiresGrad:
            bwdCode += bwdTemplate.format(ndims = ndims, idims = idims,
                                          result = result.name, inp = b.name)
        bwdTape.append(bwdCode)
        return result

add = AddFunc.apply

# add bias (1D) to 2D array
class AddBiasFunc(Func):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape[1] == b.shape[0]
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a, b))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'AddBias({idims}{a}, {b}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, idims = idims,
                                     a = a.name, b = b.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplateA = 'Add2D({idims}{result}Grad, {inp}Grad, {inp}Grad);\n'
            bwdCode += bwdTemplateA.format(idims = idims, result = result.name, inp = a.name)
        if b.requiresGrad:
            bwdTemplateB = 'AddBiasBwd({idims}{result}Grad, {inp}Grad);\n'
            bwdCode += bwdTemplateB.format(idims = idims, result = result.name, inp = b.name)
        bwdTape.append(bwdCode)
        return result

addBias = AddBiasFunc.apply

# subtract 2 tensors
class SubFunc(Func):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        ndims = len(a.shape)
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a, b))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'Subtract{ndims}D({idims}{a}, {b}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, ndims = ndims,
                                     idims = idims, a = a.name, b = b.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplateA = 'Add{ndims}D({idims}{result}Grad, {inp}Grad, {inp}Grad);\n'
            bwdCode += bwdTemplateA.format(ndims = ndims, idims = idims,
                                           result = result.name, inp = a.name)
        if b.requiresGrad:
            bwdTemplateB = 'Subtract{ndims}D({idims}{inp}Grad, {result}Grad, {inp}Grad);\n'
            bwdCode += bwdTemplateB.format(ndims = ndims, dims = idims,
                                           inp = b.name, result = result.name)
        bwdTape.append(bwdCode)
        return result

sub = SubFunc.apply

# create zero init tensor - used for hidden state init
class ZeroInitFunc(Func):
    @staticmethod
    def apply(shape: tuple) -> Tensor:
        result = Tensor(shape, False)
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'SetToZero{ndims}D({idims}{result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(shape),
                                     result = result.name,
                                     ndims = len(shape),
                                     idims = shapeToFuncCallString(shape))
        fwdTape.append(fwdCode)
        return result

zeros = ZeroInitFunc.apply

# get current timestep features from input data
class GetTimestepFunc(Func):
    @staticmethod
    def apply(a: Tensor, timestep: int) -> Tensor:
        assert len(a.shape) == 3
        result = Tensor(a.shape[1:], anyInputRequiresGrad(a))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'IndexTimestep({a}, {result}, {timestep});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(result.shape),
                                     result = result.name,
                                     a = a.name, timestep = timestep)
        fwdTape.append(fwdCode)
        if a.requiresGrad:
            raise NotImplementedError
        return result

getTimestep = GetTimestepFunc.apply

# elementwise multiply
class MulFunc(Func):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        ndims = len(a.shape)
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a, b))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'SetToZero{ndims}D({idims}{result});\n'\
                      'MAdd{ndims}D({idims}{a}, {b}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, ndims = ndims,
                                     idims = idims, a = a.name, b = b.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        bwdTemplate = 'MAdd{ndims}D({idims}{result}Grad, {other}, {inp}Grad);\n'
        if a.requiresGrad:
            bwdCode += bwdTemplate.format(ndims = ndims, idims = idims,
                                          result = result.name, other = b.name,
                                          inp = a.name)
        if b.requiresGrad:
            bwdCode += bwdTemplate.format(ndims = ndims, idims = idims,
                                          result = result.name, other = a.name,
                                          inp = b.name)
        bwdTape.append(bwdCode)
        return result

mul = MulFunc.apply

# matrix multiply
class MatrixMultFunc(Func):
    @staticmethod
    def apply(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape[1] == b.shape[0]
        result = Tensor((a.shape[0], b.shape[1]), anyInputRequiresGrad(a, b))
        allDimCallStr = shapeToFuncCallString(a.shape + b.shape[1:])
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'MatMul({alldims}{a}, {b}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(result.shape),
                                     result = result.name, alldims = allDimCallStr,
                                     a = a.name, b = b.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplateA = 'MatMulBwdA({alldims}{result}Grad, {other}, {inp}Grad);\n'
            bwdCode += bwdTemplateA.format(alldims = allDimCallStr, result = result.name,
                                           other = b.name, inp = a.name)
        if b.requiresGrad:
            bwdTemplateB = 'MatMulBwdB({alldims}{other}, {result}Grad, {inp}Grad);\n'
            bwdCode += bwdTemplateB.format(alldims = allDimCallStr, other = a.name,
                                           result = result.name, inp = b.name)
        bwdTape.append(bwdCode)
        return result

matmul = MatrixMultFunc.apply

# relu function
class ReLUFunc(Func):
    @staticmethod
    def apply(a: Tensor) -> Tensor:
        assert len(a.shape) == 2
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'ReLU2D({idims}{a}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, idims = idims,
                                     a = a.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplate = 'ReLUBwd2D({idims}{a}, {a}Grad, {result}Grad);\n'
            bwdCode = bwdTemplate.format(idims = idims, a = a.name,
                                         result = result.name)
        bwdTape.append(bwdCode)
        return result

relu = ReLUFunc.apply

# sigmoid function
class SigmoidFunc(Func):
    @staticmethod
    def apply(a: Tensor) -> Tensor:
        assert len(a.shape) == 2
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'Sigmoid2D({idims}{a}, {result}, SigmoidSegData);\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, idims = idims,
                                     a = a.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplate = 'SigmoidBwd2D({idims}{result}, {result}Grad, {a}Grad);\n'
            bwdCode = bwdTemplate.format(idims = idims, result = result.name,
                                         a = a.name)
        bwdTape.append(bwdCode)
        return result

sigmoid = SigmoidFunc.apply

# tanh function
class TanhFunc(Func):
    @staticmethod
    def apply(a: Tensor) -> Tensor:
        assert len(a.shape) == 2
        idims = shapeToFuncCallString(a.shape)
        result = Tensor(a.shape, anyInputRequiresGrad(a))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'Tanh2D({idims}{a}, {result}, TanhSegData);\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(a.shape),
                                     result = result.name, idims = idims,
                                     a = a.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplate = 'TanhBwd2D({idims}{result}, {result}Grad, {a}Grad);\n'
            bwdCode = bwdTemplate.format(idims = idims, result = result.name,
                                         a = a.name)
        bwdTape.append(bwdCode)
        return result

tanh = TanhFunc.apply

# max pool
class MaxPoolFunc(Func):
    @staticmethod
    def apply(a: Tensor, NumChan: int, InNumRows: int, InNumCols: int,
              FiltNumRows: int, FiltNumCols: int, OutNumRows: int, OutNumCols: int,
              StrideRow: int, StrideCol: int, PadRow: int, PadCol: int) -> Tensor:
        assert len(a.shape) == 2
        resultShape = (a.shape[0], str(NumChan * OutNumRows * OutNumCols))
        result = Tensor(resultShape, anyInputRequiresGrad(a))
        fwdTemplate = 'int64_al{shape} {result};\n'\
                      'MaxPool({NumSamples}, {NumChan}, {InNumRows}, {InNumCols},\n'\
                      '{FiltNumRows}, {FiltNumCols}, {OutNumRows}, {OutNumCols},\n'\
                      '{StrideRow}, {StrideCol}, {PadRow}, {PadCol},\n'\
                      '{a}, {result});\n'
        fwdCode = fwdTemplate.format(shape = shapeToArrayDefString(result.shape),
                                     result = result.name, NumSamples = a.shape[0],
                                     NumChan = NumChan, InNumRows = InNumRows,
                                     InNumCols = InNumCols, FiltNumRows = FiltNumRows,
                                     FiltNumCols = FiltNumCols, OutNumRows = OutNumRows,
                                     OutNumCols = OutNumCols, StrideRow = StrideRow,
                                     StrideCol = StrideCol, PadRow = PadRow,
                                     PadCol = PadCol, a = a.name)
        fwdTape.append(fwdCode)
        bwdCode = ''
        if a.requiresGrad:
            bwdTemplate = 'MaxPoolBwd({NumSamples}, {NumChan}, {InNumRows}, {InNumCols},\n'\
                          '{FiltNumRows}, {FiltNumCols}, {OutNumRows}, {OutNumCols},\n'\
                          '{StrideRow}, {StrideCol}, {PadRow}, {PadCol},\n'\
                          '{a}, {a}Grad, {result}, {result}Grad);\n'
            bwdCode = bwdTemplate.format(NumSamples = a.shape[0], NumChan = NumChan,
                                         InNumRows = InNumRows, InNumCols = InNumCols,
                                         FiltNumRows = FiltNumRows, FiltNumCols = FiltNumCols,
                                         OutNumRows = OutNumRows, OutNumCols = OutNumCols,
                                         StrideRow = StrideRow, StrideCol = StrideCol,
                                         PadRow = PadRow, PadCol = PadCol, a = a.name,
                                         result = result.name)
        bwdTape.append(bwdCode)
        return result

maxPool = MaxPoolFunc.apply

# mean squared error loss
class MeanSquaredErrorLossFunc(Func):
    @staticmethod
    def apply(pred: Tensor, targetName: str) -> Tensor:
        inputParam = ('int64_al' + shapeToArrayDefString(pred.shape), targetName)
        fwdInputs.append(inputParam)
        fwdTemplate = 'MSELossAndAcc({pred}, {target}, lossAndAcc);\n'
        fwdCode = fwdTemplate.format(pred = pred.name, target = targetName)
        fwdTape.append(fwdCode)
        bwdTemplate = 'MSEGrad({pred}, {target}, {pred}Grad);\n'
        bwdCode = bwdTemplate.format(pred = pred.name, target = targetName)
        bwdTape.append(bwdCode)
        return None

meanSquaredErrorLoss = MeanSquaredErrorLossFunc.apply
