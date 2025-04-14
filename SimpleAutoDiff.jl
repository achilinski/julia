# SimpleAutoDiff.jl
module SimpleAutoDiff

using Statistics, Random, LinearAlgebra

export Variable, value, grad, backward!, zero_grad!, parameters, matmul

# --- Variable Type ---

mutable struct Variable{T<:Real}
    value::Union{T, AbstractArray{T}}
    gradient::Union{Nothing, T, AbstractArray{T}}
    children::Vector{Variable} # Inputs that created this variable
    backward_fn::Function      # Function to compute gradients w.r.t children
    is_param::Bool             # Flag to identify trainable parameters

    # Constructor for leaf nodes (inputs, parameters)
    function Variable(value::Union{T, AbstractArray{T}}; is_param::Bool=false) where {T<:Real}
        grad_val = nothing
        new{T}(value, grad_val, Variable[], () -> nothing, is_param) # Use Variable[]
    end

    # Constructor for internal nodes (results of operations)
    function Variable(value::Union{T, AbstractArray{T}}, children::Vector{Variable}, backward_fn::Function) where {T<:Real}
        grad_val = nothing
        new{T}(value, grad_val, children, backward_fn, false) # Results of ops are not params by default
    end
end

# --- Helper Functions ---

value(v::Variable) = v.value
value(x::Real) = x # Allow mixing Variables and constants

grad(v::Variable) = v.gradient

# Use element type of the variable's value for type safety
_eltype(v::Variable{T}) where T = T
_eltype(v::AbstractArray{T}) where T = T
_eltype(v::T) where T<:Real = T

function grad!(v::Variable{T}, g) where T # Capture T here
    # Ensure assigned gradient matches the variable's type T
    g_converted = if isa(g, AbstractArray)
        convert(AbstractArray{T}, g)
    else
        convert(T, g) # Ensure scalar matches T
    end

    if v.gradient === nothing
        # If g is scalar (like one(T)) and v.value is array, fill the gradient array
# --- START CHANGE ---
        if isa(v.value, AbstractArray) && isa(g_converted, Real)
            # Initialize gradient as an array filled with the scalar value g_converted
            v.gradient = fill(g_converted, size(value(v)))
        else # Otherwise, assume shapes match or g is array matching v.value, or both are scalar
            # Use deepcopy to avoid aliasing issues with the input g
            v.gradient = deepcopy(g_converted)
        end
# --- END CHANGE ---
    else
         # Let accumulate handle addition if already initialized
        accumulate_gradient!(v, g_converted) # Pass converted grad
    end
end

# Function to collect all parameters
function parameters(v::Variable)
    params = Set{Variable}()
    visited = Set{Variable}()
    nodes_to_visit = [v]
    while !isempty(nodes_to_visit)
        current = pop!(nodes_to_visit)
        if !(current in visited)
            push!(visited, current)
            if current.is_param; push!(params, current); end
            for child in current.children; push!(nodes_to_visit, child); end
        end
    end
    return collect(params) # Return as Vector{Variable}
end

# Nicer printing showing type
Base.show(io::IO, v::Variable) = print(io, "Variable{$(eltype(v.value))}(grad=$(v.gradient !== nothing))")

# --- Gradient Accumulation ---
function accumulate_gradient!(v::Variable{T}, g) where {T<:Real}
    # Ensure incoming gradient `g` is converted to type T before accumulation
    g_converted = if isa(g, AbstractArray)
        convert(AbstractArray{T}, g)
    elseif isa(g, Real)
         convert(T, g)
    else
         g # Should ideally not happen if called correctly
    end

    if v.gradient === nothing
        v.gradient = deepcopy(g_converted)
    else
        # Check for shape compatibility before accumulating
        if size(v.gradient) == size(g_converted)
            v.gradient .+= g_converted
        elseif isa(v.gradient, AbstractArray) && isa(g_converted, Real)
            v.gradient .+= g_converted # Broadcast scalar update
        elseif isa(v.gradient, Real) && isa(g_converted, AbstractArray)
            v.gradient += sum(g_converted) # Sum array update to scalar grad
        else
            try # Try broadcasting accumulation if sizes differ but might be compatible
                v.gradient .+= g_converted
            catch e
                 println("Warning: Gradient accumulation size mismatch. grad: $(size(v.gradient)), update: $(size(g_converted))")
                 isa(v.gradient, Real) ? (v.gradient += sum(g_converted)) : error("Cannot accumulate gradients of incompatible sizes $(size(v.gradient)) and $(size(g_converted))")
            end
        end
    end
end

# --- Backpropagation ---
function backward!(v::Variable{T}) where {T<:Real}
    # Initialize gradient of the final node (loss) to 1.0 of the correct type T
    # Allow starting if value is scalar OR a single-element array
   if v.gradient === nothing
# --- START CHANGE ---
       # Check if the value is a scalar Real OR an array with exactly one element
       if isa(v.value, Real) || length(v.value) == 1
           # Initialize gradient to one(T). grad! will handle filling an array if v.value is array.
           grad!(v, one(T))
       else
           # Error if it's a multi-element array without a pre-set gradient
           error("Backward! started on non-scalar, multi-element Variable without initial gradient. Shape: $(size(v.value)), Type: $T")
       end
# --- END CHANGE ---
   end

   # --- Topological Sort and Processing ---
   topo_stack = Variable[]
   visited_topo = Set{Variable}()
   function build_topo_stack(node)
        push!(visited_topo, node)
        for child in node.children
             if !(child in visited_topo); build_topo_stack(child); end
        end
        push!(topo_stack, node) # Add node after visiting inputs
   end
   build_topo_stack(v)

   # Process stack in reverse topological order
   visited_in_pass = Set{Variable}()
   while !isempty(topo_stack)
       current_node = pop!(topo_stack)
       if current_node.gradient !== nothing && !(current_node in visited_in_pass)
           current_node.backward_fn() # Propagate/accumulate gradients to inputs (children)
           push!(visited_in_pass, current_node)
       end
   end
end

# --- Zero Gradients ---
function zero_grad!(params::AbstractVector{<:Variable})
    for p in params
        T = _eltype(p.value) # Get the type T (e.g., Float32)
        if p.gradient !== nothing
            p.gradient = isa(p.gradient, AbstractArray) ? (p.gradient .= zero(T); p.gradient) : zero(T)
        end
        # If gradient is nothing, it remains nothing
    end
end

# --- Helper to create Variables from Reals, matching another Variable's type ---
# function _var_like(ref::Variable{T}, val::Real) where T
#     Variable(fill(T(val), size(value(ref))), is_param=false)
# end

# --- Overloaded Operations (Ensuring Type T Consistency and Children Vector) ---

# Define local new_var pattern for closure


# Addition
function Base.:+(a::Variable{T}, b::Variable{T}) where {T<:Real}
    val = value(a) .+ value(b)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a = output_grad
        grad_b = output_grad
         if size(value(a)) != size(output_grad); grad_a = sum_to(output_grad, size(value(a))); end
         if size(value(b)) != size(output_grad); grad_b = sum_to(output_grad, size(value(b))); end
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:+(a::Variable{T}, b::Real) where T = a + Variable(fill(T(b), size(value(a))))
Base.:+(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) + b

# Subtraction
function Base.:-(a::Variable{T}, b::Variable{T}) where {T<:Real}
    val = value(a) .- value(b)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a = output_grad
        grad_b = -output_grad # Negate gradient for subtraction
         if size(value(a)) != size(output_grad); grad_a = sum_to(output_grad, size(value(a))); end
         if size(value(b)) != size(output_grad); grad_b = sum_to(-output_grad, size(value(b))); end # Apply negation before sum_to
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:-(a::Variable{T}, b::Real) where T = a - Variable(fill(T(b), size(value(a))))
Base.:-(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) - b

# Unary minus
function Base.:-(a::Variable{T}) where {T<:Real}
    zero_val = zeros(T, size(value(a)))
    zero_var = Variable(zero_val, is_param=false)
    return zero_var - a # Reuse binary subtraction
end

# Element-wise Multiplication
function Base.:*(a::Variable{T}, b::Variable{T}) where {T<:Real}
     # Dispatch to matmul if appropriate shapes
     if isa(value(a), AbstractMatrix) && isa(value(b), AbstractMatrix)
         return matmul(a,b)
     elseif isa(value(a), AbstractVecOrMat) && isa(value(b), AbstractVector) && size(value(a), 2) == length(value(b))
         # Potential Mat-Vec case, treat vector as column
         # Reshape b carefully if needed by matmul implementation
         println("Warning: Using '*' for potential Mat-Vec. Consider explicit matmul. Assuming element-wise or broadcast.")
     elseif isa(value(a), AbstractVector) && isa(value(b), AbstractVecOrMat) && length(value(a)) == size(value(b), 1)
         println("Warning: Using '*' for potential Vec-Mat. Consider explicit matmul. Assuming element-wise or broadcast.")
     end

     # Default to element-wise / broadcasting
    val = value(a) .* value(b)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a_unshaped = output_grad .* value(b)
        grad_b_unshaped = output_grad .* value(a)
        grad_a = sum_to(grad_a_unshaped, size(value(a)))
        grad_b = sum_to(grad_b_unshaped, size(value(b)))
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:*(a::Variable{T}, b::Real) where T = a * Variable(fill(T(b), size(value(a))))
Base.:*(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) * b

# Matrix Multiplication
function matmul(a::Variable{T}, b::Variable{T}) where T
    # Basic dimension check
    size_a = size(value(a))
    size_b = size(value(b))
    if length(size_a) != 2 || length(size_b) != 2 || size_a[2] != size_b[1]
        error("Incompatible matrix dimensions for matmul: $(size_a) and $(size_b)")
    end

    val = value(a) * value(b) # LinearAlgebra.*
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var) # dL/dy
        grad_a = output_grad * transpose(value(b)) # dL/da = dL/dy * b'
        grad_b = transpose(value(a)) * output_grad # dL/db = a' * dL/dy
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end

# Division (element-wise)
function Base.:/(a::Variable{T}, b::Variable{T}) where {T<:Real}
    # Add epsilon for stability during division and gradient calculation
    eps_T = Base.eps(T)
    val = value(a) ./ (value(b) .+ eps_T)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        denom_stable = value(b) .+ eps_T # Use the same stable denominator
        grad_a_unshaped = output_grad ./ denom_stable
        grad_b_unshaped = -output_grad .* value(a) ./ (denom_stable .^ 2)
        grad_a = sum_to(grad_a_unshaped, size(value(a)))
        grad_b = sum_to(grad_b_unshaped, size(value(b)))
        accumulate_gradient!(a, grad_a)
        accumulate_gradient!(b, grad_b)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:/(a::Variable{T}, b::Real) where T = a / Variable(fill(T(b), size(value(a))))
Base.:/(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) / b

# Power (element-wise)
function Base.:^(a::Variable{T}, n::Real) where {T<:Real}
    n_T = T(n) # Convert exponent to type T
    # Add stability for gradient if base can be zero or negative
    eps_T = Base.eps(T)
    base_stable = value(a) .+ T(sign(value(a)) * eps_T) # Add small epsilon away from zero
    val = base_stable .^ n_T
    children = Variable[a]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a_unshaped = output_grad .* n_T .* (base_stable .^ (n_T - one(T)))
        grad_a = sum_to(grad_a_unshaped, size(value(a))) # Sum back to original shape
        accumulate_gradient!(a, grad_a)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end

# Exponential (element-wise)
function Base.exp(a::Variable{T}) where {T<:Real}
    val = exp.(value(a))
    children = Variable[a]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a_unshaped = output_grad .* val # d(exp(x))/dx = exp(x)
        grad_a = sum_to(grad_a_unshaped, size(value(a)))
        accumulate_gradient!(a, grad_a)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end

# Logarithm (element-wise)
function Base.log(a::Variable{T}; ϵ::Union{Nothing,Real}=nothing) where {T<:Real}
     eps_T = ϵ === nothing ? Base.eps(T) : T(ϵ)
     val_stable = max.(value(a), eps_T) # Ensure input >= eps_T
     val = log.(val_stable)
     children = Variable[a]
     local new_var
    function backward_fn()
        output_grad = grad(new_var)
        grad_a_unshaped = output_grad ./ val_stable # Use stable value for gradient
        grad_a = sum_to(grad_a_unshaped, size(value(a)))
        accumulate_gradient!(a, grad_a)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end

# Max (element-wise max with a scalar)
function Base.max(a::Variable{T}, val::Real) where {T<:Real}
    val_T = T(val) # Convert scalar to type T
    res_val = max.(value(a), val_T)
    children = Variable[a]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        mask = T.(value(a) .> val_T) # Gradient flows only where a > val_T
        grad_a_unshaped = output_grad .* mask
        grad_a = sum_to(grad_a_unshaped, size(value(a)))
        accumulate_gradient!(a, grad_a)
    end
    new_var = Variable(res_val, children, backward_fn)
    return new_var
end
# Define the other direction for completeness
Base.max(val::Real, a::Variable{T}) where T = max(a, val) # Reuse existing


# Sum (reduce to scalar)
function Base.sum(a::Variable{T}) where {T<:Real}
    val = sum(value(a)) # Result is scalar T
    children = Variable[a]
    local new_var
    function backward_fn()
        output_grad = grad(new_var) # Should be scalar T
        grad_a = fill(output_grad, size(value(a))) # Broadcast scalar grad back to input shape
        accumulate_gradient!(a, grad_a)
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end

# Helper for broadcasting gradients back to original shape
function sum_to(x::AbstractArray{T}, target_size::Tuple) where T
    if size(x) == target_size; return x; end
    if isempty(target_size) || target_size == (1,); return sum(x)::T; end # Target is scalar

    ndims_x = ndims(x)
    ndims_target = length(target_size)
    dims_to_sum = Int[]
    for d = 1:ndims_x
        if d > ndims_target || (target_size[d] == 1 && size(x, d) > 1)
            push!(dims_to_sum, d)
        elseif d <= ndims_target && target_size[d] != 1 && size(x, d) != target_size[d] && size(x, d) != 1
             error("Cannot sum_to: Incompatible shapes $(size(x)) to $(target_size) along dimension $d")
        end
    end

    result = isempty(dims_to_sum) ? x : sum(x, dims=tuple(dims_to_sum...))
    # After sum, result might have dimensions of size 1. Reshape to strictly match target.
    return size(result) == target_size ? result : reshape(result, target_size)
end
function sum_to(x::T, target_size::Tuple) where T<:Real # Scalar input gradient
    return fill(x, target_size)::AbstractArray{T}
end

end # module SimpleAutoDiff