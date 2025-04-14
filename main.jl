# main.jl

println("Loading modules...")

# Include the files first to define the modules
include("SimpleAutoDiff.jl")
include("MLP.jl")
include("Loss.jl")
include("Optimizer.jl")
include("DataProcessing.jl")


# Now use 'using' to bring the *defined* modules into scope
using .SimpleAutoDiff
using .MLPDefinition
using .LossFunctions
using .Optimizers
using .DataProcessing

# Standard libraries
using Random, Statistics, Dates, JLD2

println("Modules loaded and included.")

# --- Configuration ---
println("Setting up configuration...")
# Use explicit Float32 for learning rate if desired (0.01f0)
# Otherwise Float64 (0.01) is fine IF the optimizer handles conversion or is parametric
const JLD2_FILE = "imdb_dataset.jld2"
const VOCAB_MIN_FREQ = 5
const VOCAB_MAX_WORDS = 10000
const HIDDEN_SIZE = 32
const OUTPUT_SIZE = 1
const LEARNING_RATE = 0.01 # Float64 okay here, will be converted
const EPOCHS = 5
const TARGET_DTYPE = Float32 # Define the target type for parameters and optimizer

# --- Create Dummy Data if file not found ---
function create_dummy_data(filepath, num_samples=100)
     if !isfile(filepath)
         println("'$filepath' not found. Creating dummy data...")
         dummy_reviews = ["this movie was good " ^ rand(5:10) * rand(["great ", "ok ", "bad "]) for _ in 1:num_samples]
         dummy_labels = rand(Bool, num_samples)
         try
             jldsave(filepath; reviews=dummy_reviews, labels=dummy_labels)
             println("Dummy data saved to '$filepath'.")
         catch e; println("Error creating dummy data: $e"); end
     end
 end

# --- Main Training Logic ---
function main()
    println("Starting training script...")
    create_dummy_data(JLD2_FILE) # Create dummy if needed

    # 1. Load and Preprocess Data (DataProcessing ensures Float32)
    println("Preprocessing data...")
    train_data, test_data, vocab, input_size = DataProcessing.preprocess_data(
        JLD2_FILE, min_freq=VOCAB_MIN_FREQ, max_words=VOCAB_MAX_WORDS
    )

    if isempty(train_data) || input_size == 0
        println("Failed to load or process data. Exiting.")
        return
    end
    println("Data preprocessed. Input dimension (vocab size): $input_size")

    # Determine parameter type from data (should be Float32)
    # We assume all parameters will use this type. TARGET_DTYPE could enforce this.
    param_dtype = TARGET_DTYPE # Use the globally defined target type

    # 2. Build MLP Model (Pass dtype explicitly)
    println("Building MLP model...")
    model = MLPDefinition.MLPModel(
        MLPDefinition.Dense(input_size, HIDDEN_SIZE, MLPDefinition.relu, dtype=param_dtype),
        MLPDefinition.Dense(HIDDEN_SIZE, OUTPUT_SIZE, MLPDefinition.sigmoid, dtype=param_dtype)
    )
    model_params = MLPDefinition.get_params(model)
    println("Model created with $(length(model_params)) parameter tensors of type $param_dtype.")

    # 3. Optimizer (Create with the correct type T)
    # Convert LEARNING_RATE to the parameter data type
    lr_T = param_dtype(LEARNING_RATE)
    # Ensure params vector is correctly typed for the optimizer constructor
    # typed_params = convert(Vector{Variable{param_dtype}}, model_params) # Old way
    typed_params = Variable{param_dtype}[p for p in model_params] # Construct typed vector
    optimizer = Optimizers.SGD(lr_T, typed_params) # Use parametric SGD
    println("Optimizer SGD created with LR=$(optimizer.lr) of type $(typeof(optimizer.lr)).")

    # 4. Training Loop
    println("\n--- Starting Training ---")
    total_train_samples = length(train_data)

    for epoch in 1:EPOCHS
        epoch_start_time = now()
        # total_loss = 0.0 # Was Float64
        # Initialize with zero of the correct parameter type
        total_loss = zero(param_dtype)
        samples_processed = 0
        correct_predictions = 0

        # Shuffle training data each epoch
        shuffle!(train_data)

        for (i, (x_vec, y_true)) in enumerate(train_data) # x_vec::Vector{Float32}, y_true::Float32
            # Ensure input Variable matches parameter type
            x = SimpleAutoDiff.Variable(reshape(x_vec, 1, :), is_param=false) # Should be Variable{Float32}

            # --- Forward Pass ---
            y_pred = model(x) # Should return Variable{Float32}

            # --- Calculate Loss ---
            # Pass Float32 y_true to loss function expecting Real
            loss = LossFunctions.binary_cross_entropy(y_pred, y_true) # Should return scalar Variable{Float32} (value is likely Matrix{Float32}(1,1))

            # total_loss += SimpleAutoDiff.value(loss) # ERROR: Float64 + Matrix{Float32}
            # Extract the scalar value from the loss Variable's underlying value (which might be a 1x1 matrix)
            loss_scalar_value = SimpleAutoDiff.value(loss)[1] # Get the first (and only) element
            total_loss += loss_scalar_value # Now Float32 + Float32

            samples_processed += 1

            # --- Backward Pass ---
            SimpleAutoDiff.zero_grad!(optimizer.params) # Zeros gradients based on param type
            SimpleAutoDiff.backward!(loss) # Computes gradients (should be Float32)

            # --- Update Parameters ---
            Optimizers.update!(optimizer) # Updates params in-place using type T

            # --- Track Accuracy ---
            pred_label = SimpleAutoDiff.value(y_pred)[1] >= 0.5 ? param_dtype(1.0) : param_dtype(0.0)
            # Compare predicted label (Float32) with true label (Float32)
            if isapprox(pred_label, y_true) # Use isapprox for float comparison
                correct_predictions += 1
            end

             # Print progress occasionally
             if i % max(1, total_train_samples ÷ 10) == 0 || i == total_train_samples
                  progress = round(100 * i / total_train_samples, digits=1)
                  avg_loss_print = total_loss / samples_processed
                  accuracy = round(100 * correct_predictions / samples_processed, digits=2)
                  print("\rEpoch $epoch [$progress%] Avg Loss: $(round(avg_loss_print, digits=4)), Acc: $accuracy%")
             end
        end # End batch loop

        epoch_duration = now() - epoch_start_time
        final_avg_loss = total_loss / samples_processed
        final_accuracy = round(100 * correct_predictions / samples_processed, digits=2)
        println("\nEpoch $epoch completed in $epoch_duration. Avg Loss: $(round(final_avg_loss, digits=4)), Train Accuracy: $final_accuracy%")

        # Optional: Evaluate on test set
        # Ensure evaluate function also respects types
        test_acc, test_loss = evaluate(model, test_data, param_dtype)
        println("Test Loss: $(round(test_loss, digits=4)), Test Accuracy: $(round(test_acc*100, digits=2))%")

    end # End epoch loop

    println("--- Training Finished ---")
end

# --- Evaluation Function (Optional) ---
function evaluate(model::MLPDefinition.MLPModel, test_data::Vector, param_dtype::Type{<:Real})
    # total_loss = 0.0 # Was Float64
    # Initialize with zero of the correct type
    total_loss = zero(param_dtype)
    correct_predictions = 0
    num_samples = length(test_data)
    if num_samples == 0; return (param_dtype(0.0), param_dtype(0.0)); end # Return correct type

    for (x_vec, y_true) in test_data # x_vec::Vector{Float32}, y_true::Float32
         # Ensure Variable matches model parameter type
         x = SimpleAutoDiff.Variable(reshape(x_vec, 1, :), is_param=false) # Variable{Float32}
         y_pred = model(x) # Forward pass only -> Variable{Float32}
         loss = LossFunctions.binary_cross_entropy(y_pred, y_true) # -> scalar Variable{Float32} (value maybe Matrix)

         # total_loss += SimpleAutoDiff.value(loss) # Potential Error: Float + Matrix
         # Extract the scalar value from the loss Variable's underlying value
         loss_scalar_value = SimpleAutoDiff.value(loss)[1]
         total_loss += loss_scalar_value # Now Float32 + Float32

         pred_label = SimpleAutoDiff.value(y_pred)[1] >= 0.5 ? param_dtype(1.0) : param_dtype(0.0)
         if isapprox(pred_label, y_true) # Use isapprox for float comparison
             correct_predictions += 1
         end
    end

    avg_loss = total_loss / num_samples # Float32 / Int -> Float32 or Float64 depending on Julia version/settings
    accuracy = correct_predictions / num_samples # Int / Int -> Float64
    # Cast results if needed, but usually Float64 is fine for final reporting
    return accuracy, avg_loss # Return Float accuracy, Float32/Float64 loss
end


# --- Run ---
if abspath(PROGRAM_FILE) == @__FILE__
    main()
else
    println("Script included, not running main(). Call main() manually if needed.")
end