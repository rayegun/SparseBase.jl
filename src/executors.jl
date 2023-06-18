abstract type Executor end

# need notion of options.
# this is mostly here for the future, need to hook up Finch and GrB
# before this becomes very interesting

# TODO: Finish hooking up GrBExecutor for SSGrB
# TODO: test matmul for FinchExecutor.
# TODO: Options, which will probably need to be barriered into type domain.

struct CUDAExecutor <: Executor end
struct AMDExecutor <: Executor end
struct MKLExecutor <: Executor end
struct GrBExecutor <: Executor end
struct FinchExecutor <: Executor end
struct TACOExecutor <: Executor end
struct SpartanExecutor <: Executor end # by hand Spartan impls

struct ReferenceExecutor <: Executor end # Basic executor meant to prove the correctness of other executors.
struct DefaultExecutor <: Executor end # meh, this probably shouldn't exist
# might be best for capturing ::SparsesBase.Executor
struct AutoExecutor <: Executor end # some autoselection method.

function getexecutor(f, args...)
    return combineexecutors(f, getexecutor.(args)...)
end

getexecutor(arg) = DefaultExecutor()

# Todo: requires preferenes and such before actual impl.

combineexecutors(f) = DefaultExecutor()
combineexecutors(f, e) = e
combineexecutors(f, e1::E, e2::E) where {E} = E()
function combineexecutors(f, e1, e2, es...)
    return combineexecutors(combineexecutors(f, e1), combineexecutors(f, e2, es...))
end

# need rules
