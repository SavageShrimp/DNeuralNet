/**
 * Neural Network Implementation in D
 * This code provides a basic framework for a neural network with layers, activation functions, and parameter updates.
 * https://arxiv.org/pdf/2503.10622
 */

import std.random: uniform, uniform01;
import std.math: exp, abs, sqrt, pow, log, tanh;
import std.range: zip;
import std.stdio;
import std.format;
import std.algorithm;
import std.conv: to;
import std.array;
import std.exception;
import std.file;

//public import neuralnet.netcore;
//public import neuralnet.layeractivation;

enum LayerType {
    Unused,
    Dense,
    ActivationSigmoid,
    LayerNorm,
    DynamicTanh,
    ErrorLayer,
    ActivationSoftmax,
    DenseL2,
    ActivationReLU,
    ActivationLeakyReLU,
    ActivationTanh
}


class NetworkTopology {
    size_t[] topology;
    real[] params;

    void serialize(File file) {
        file.rawWrite([topology.length]);
        file.rawWrite(topology);
        file.rawWrite([params.length]);
        file.rawWrite(params);
    }

    void deserialize(File file) {
        size_t[1] topoLength;
        file.rawRead(topoLength);
        topology.length = topoLength[0];
        file.rawRead(topology);

        size_t[1] paramLength;
        file.rawRead(paramLength);
        params.length = paramLength[0];
        file.rawRead(params);
    }
}

/**
 * Checks if two real numbers are approximately equal.
 * Parameters:
 * - a: The first real number.
 * - b: The second real number.
 * - epsilon: The tolerance level for the comparison.
 * Returns:
 * - true if the absolute difference between `a` and `b` is less than `epsilon`, false otherwise.
 */
bool approxEqual(real a, real b, real epsilon = 1e-6) {
    return abs(a - b) < epsilon;
}

/**
 * Interface for a layer in the neural network.
 * Provides methods for forward and backward operations.
 */
interface Layer {
    /**
     * Forward pass through the layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    real[] forward(real[] input);

    /**
     * Backward pass through the layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    real[] backward(real[] gradient);
}

/**
 * Interface for a layer with parameters (weights and biases).
 * Extends the Layer interface with an update method.
 */
interface ParameterLayer : Layer {
    /**
     * Updates the layer's parameters based on the given learning rate.
     * Parameters:
     * - learningRate: The learning rate for the update.
     */
    void update(real learningRate);
}



class Parameters {
    real[] weights;
    real[] biases;
    real[] weightGrads;
    real[] biasGrads;

    this(size_t inputSize, size_t outputSize) {
        initializeWeights(weights, inputSize, outputSize);
        biases.length = outputSize;
        biases[] = 0.0;
        weightGrads.length = weights.length;
        weightGrads[] = 0.0;
        biasGrads.length = outputSize;
        biasGrads[] = 0.0;
    }

    void update(real learningRate) {
        foreach (i; 0 .. weights.length) {
            weights[i] -= learningRate * weightGrads[i];
        }
        foreach (i; 0 .. biases.length) {
            biases[i] -= learningRate * biasGrads[i];
        }
        weightGrads[] = 0.0;
        biasGrads[] = 0.0;
    }

    void serialize(NetworkTopology topology) {
        topology.params ~= weights;
        topology.params ~= biases;
    }

    void deserialize(NetworkTopology topology, size_t inputSize, size_t outputSize) {
        size_t weightsStart = 0;
        size_t weightsEnd = outputSize * inputSize;
        weights = topology.params[weightsStart .. weightsEnd];
        size_t biasesStart = weightsEnd;
        size_t biasesEnd = biasesStart + outputSize;
        biases = topology.params[biasesStart .. biasesEnd];
    }

    private void initializeWeights(ref real[] weights, size_t inputSize, size_t outputSize) {
        weights.length = outputSize * inputSize;
        double g = sqrt(6.0 / (inputSize + outputSize));
        foreach (i; 0 .. weights.length) {
            weights[i] = uniform(-1.0, 1.0) * g;
        }
    }

}


interface Serializable {
    /**
     * Serializes the layer's data and sizes into a string or binary format.
     * Returns:
     * - The serialized data as a string.
     */
    void serialize(NetworkTopology topology);

    /**
     * Deserializes the layer's data and sizes from a string or binary format.
     * Parameters:
     * - data: The serialized data as a string.
     */
    void deserialize(NetworkTopology topology);
}



version(unused)
{
void initializeWeights(ref real[] weights, size_t inputSize, size_t outputSize) {
    weights.length = outputSize * inputSize;
    double g = sqrt(6.0 / (inputSize + outputSize));
    foreach (i; 0 .. weights.length) {
//        weights[i] = uniform(-g, g);
        weights[i] = uniform(-1.0, 1.0);
    }
}
}


class Dense: ParameterLayer, Serializable {
    Parameters params;
    real[] input;
    real[] output;
    real[] nextGradient;

    size_t inputSize;
    size_t outputSize;

    this(size_t inputSize, size_t outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        params = new Parameters(inputSize, outputSize);
    }

    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output.length = outputSize;
        foreach (i; 0 .. outputSize) {
            output[i] = params.biases[i];
            foreach (j; 0 .. inputSize) {
                output[i] += params.weights[i * inputSize + j] * input[j];
            }
        }
        return output;
    }

    override real[] backward(real[] gradient) {
        nextGradient.length = inputSize;
        nextGradient[] = 0.0;

        foreach (i; 0 .. outputSize) {
            params.biasGrads[i] += gradient[i];
            foreach (j; 0 .. inputSize) {
                params.weightGrads[i * inputSize + j] += gradient[i] * input[j];
                nextGradient[j] += params.weights[i * inputSize + j] * gradient[i];
            }
        }

        return nextGradient;
    }

    override void update(real learningRate) {
        params.update(learningRate);
    }

    void serialize(NetworkTopology topology) {
        topology.topology ~= cast(size_t)LayerType.Dense;
        topology.topology ~= inputSize;
        topology.topology ~= outputSize;
        params.serialize(topology);
    }

    void deserialize(NetworkTopology topology) {
        if (topology.topology[0] != cast(size_t)LayerType.Dense) {
            throw new Exception("Layer type mismatch");
        }
        inputSize = topology.topology[1];
        outputSize = topology.topology[2];
        params.deserialize(topology, inputSize, outputSize);
    }
}


/**
 * Represents a layer that applies the sigmoid activation function.
 * Implements the Layer interface.
 */
class ActivationSigmoid : Layer, Serializable {
    real[] input;
    real[] output;

    /**
     * Forward pass through the Sigmoid activation layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output = new real[input.length];

        foreach (i, val; input) {
            output[i] = 1.0 / (1.0 + exp(-val));
        }

        return output;
    }

    /**
     * Backward pass through the Sigmoid activation layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        real[] sigmoidDerivative = new real[input.length];
        foreach (i, val; output) {
            sigmoidDerivative[i] = val * (1.0 - val);
        }

        real[] nextGradient = new real[input.length];
        foreach (i, val; gradient) {
            nextGradient[i] = val * sigmoidDerivative[i];
        }
        return nextGradient;
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.ActivationSigmoid;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ActivationSigmoid) {
            throw new Exception("Layer type mismatch");
        }
    }
}

/**
 * Represents a Layer Normalization layer.
 * Implements the ParameterLayer interface.
 */
class LayerNorm : ParameterLayer, Serializable {
    real[] gamma;
    real[] beta;
    real[] input;
    real[] normalized;
    real mean;
    real variance;
    real epsilon = 1e-5;
    real[] gammaGrads;
    real[] betaGrads;

    /**
     * Constructor initializes the scaling (gamma) and shifting (beta) parameters and their respective gradients.
     * Parameters:
     * - featureSize: The size of the input to the layer.
     */
    this(size_t featureSize) {
        gamma.length = featureSize; gamma[] = 1.0;
        beta.length = featureSize; beta[] = 0.0;
        gammaGrads.length = featureSize; gammaGrads[] = 0.0;
        betaGrads.length = featureSize; betaGrads[] = 0.0;
    }

    /**
     * Forward pass through the Layer Normalization layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        real meanVal = input.sum / input.length;
        real varianceVal = input.map!(x => pow(x - meanVal, 2)).sum / input.length;
        real stdVal = sqrt(varianceVal + epsilon);

        normalized = new real[input.length];
        foreach (i, x; input) {
            normalized[i] = (x - meanVal) / stdVal;
        }

        real[] output = new real[input.length];
        foreach (i, val; normalized) {
            output[i] = gamma[i] * val + beta[i];
        }

        mean = meanVal;
        variance = varianceVal;
        return output.dup;
    }

    /**
     * Backward pass through the Layer Normalization layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        size_t m = input.length;
        real inv_std = 1 / sqrt(variance + epsilon);

        foreach (i, g; gradient) {
            betaGrads[i] += g;
            gammaGrads[i] += g * normalized[i];
        }

        real[] dx_hat = new real[m];
        foreach (i, g; gradient) {
            dx_hat[i] = gamma[i] * g;
        }

        real sum_dx_hat = dx_hat.sum;
        real sum_dxhat_xhat = 0.0;
        foreach (size_t i, val; normalized) {
            sum_dxhat_xhat += val * dx_hat[i];
        }

        real[] numerator = new real[m];
        foreach (size_t i, dx; dx_hat) {
            numerator[i] = m * dx - sum_dx_hat - normalized[i] * sum_dxhat_xhat;
        }

        real[] dx = new real[m];
        foreach (size_t i, num; numerator) {
            dx[i] = (inv_std / m) * num;
        }

        return dx;
    }

    /**
     * Updates the gamma and beta parameters of the Layer Normalization layer.
     * Parameters:
     * - learningRate: The learning rate for the update.
     */
    override void update(real learningRate) {
        foreach (i, g; gammaGrads) {
            gamma[i] -= learningRate * g;
            beta[i] -= learningRate * betaGrads[i];
        }
        gammaGrads[] = 0.0;
        betaGrads[] = 0.0;
    }

    void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.LayerNorm;
        // Store the number of elements and dimensions
        topology.topology ~= gamma.length;
        // Store the parameters
        topology.params ~= gamma;
        topology.params ~= beta;
    }

    void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.LayerNorm) {
            throw new Exception("Layer type mismatch");
        }

        // Retrieve the dimensions
        size_t featureSize = topology.topology[1];

        // Retrieve the parameters
        size_t gammaStart = 0;
        size_t gammaEnd = featureSize;
        size_t betaStart = gammaEnd;
        size_t betaEnd = betaStart + featureSize;

        gamma = topology.params[gammaStart .. gammaEnd];
        beta = topology.params[betaStart .. betaEnd];
    }
}


class DynamicTanh : ParameterLayer, Serializable {
    real alpha;
    real[] gamma;
    real[] beta;
    real[] input;
    real[] tanhOutput;
    real[] alphaGrads;
    real[] gammaGrads;
    real[] betaGrads;

    this(size_t featureSize, real initAlpha = 0.5) {
        alpha = initAlpha;
        gamma.length = featureSize; gamma[] = 1.0;
        beta.length = featureSize; beta[] = 0.0;
        alphaGrads.length = 1; alphaGrads[] = 0.0;
        gammaGrads.length = featureSize; gammaGrads[] = 0.0;
        betaGrads.length = featureSize; betaGrads[] = 0.0;
    }

    override real[] forward(real[] input_v) {
        input = input_v.dup;
        tanhOutput = new real[input.length];
        foreach (i, x; input) {
            tanhOutput[i] = tanh(alpha * x);
        }

        real[] output = new real[input.length];
        foreach (i, val; tanhOutput) {
            output[i] = gamma[i] * val + beta[i];
        }

        return output.dup;
    }

    override real[] backward(real[] gradient) {
        size_t m = input.length;

        foreach (i, g; gradient) {
            betaGrads[i] += g;
            gammaGrads[i] += g * tanhOutput[i];
            alphaGrads[0] += g * gamma[i] * (1 - pow(tanhOutput[i], 2)) * input[i];
        }

        real[] dx = new real[m];
        foreach (i, g; gradient) {
            dx[i] = g * gamma[i] * (1 - pow(tanhOutput[i], 2)) * alpha;
        }

        return dx;
    }

    override void update(real learningRate) {
        alpha -= learningRate * alphaGrads[0];
        foreach (i, g; gammaGrads) {
            gamma[i] -= learningRate * g;
            beta[i] -= learningRate * betaGrads[i];
        }
        alphaGrads[0] = 0.0;
        gammaGrads[] = 0.0;
        betaGrads[] = 0.0;
    }

    void serialize(NetworkTopology topology) {
        topology.topology ~= cast(size_t)LayerType.DynamicTanh;
        topology.topology ~= gamma.length;
        topology.params ~= alpha;
        topology.params ~= gamma;
        topology.params ~= beta;
    }

    void deserialize(NetworkTopology topology) {
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.DynamicTanh) {
            throw new Exception("Layer type mismatch");
        }

        size_t featureSize = topology.topology[1];

        alpha = topology.params[0];
        size_t gammaStart = 1;
        size_t gammaEnd = gammaStart + featureSize;
        size_t betaStart = gammaEnd;
        size_t betaEnd = betaStart + featureSize;

        gamma = topology.params[gammaStart .. gammaEnd];
        beta = topology.params[betaStart .. betaEnd];
    }
}

unittest {
    auto dt = new DynamicTanh(3);
    real[] input = [0.0, 1.0, -1.0];
    auto output = dt.forward(input);
    writeln("Output: ", output);
    real[] gradient = [1.0, 1.0, 1.0];
    auto dy_dinput = dt.backward(gradient);
    writeln("Gradient: ", dy_dinput);
    dt.update(0.1);
}

/**
 * Represents an Error layer used to calculate the gradient of the loss function.
 * Implements the Layer interface.
 */
class ErrorLayer : Layer, Serializable {
    real[] target;
    real[] prediction;

    /**
     * Forward pass through the Error layer, setting the target.
     * Parameters:
     * - input: The input data for the layer (target).
     * Returns:
     * - The input itself (target).
     */
    override real[] forward(real[] input) {
        target = input.dup;
        //writeln(i"target length = $(target.length)");
        return input;
    }

    /**
     * Backward pass through the Error layer, calculating the derivative of the loss function.
     * Parameters:
     * - gradient: The gradient from the next layer (not used here).
     * Returns:
     * - The derivative w.r.t. the prediction.
     */
    override real[] backward(real[] gradient) {
        // Calculate the derivative of the loss function w.r.t. the prediction
        // For Mean Squared Error: Loss = 0.5 * sum((prediction - target)^2)
        // The derivative w.r.t. the prediction is: prediction - target
        real[] derivative = new real[target.length];
        foreach (i, val; target) {
            derivative[i] = prediction[i] - val;
        }
        return derivative;
    }

    /**
     * Sets the prediction for the Error layer.
     * Parameters:
     * - prediction: The prediction from the network.
     */
    void setPrediction(real[] prediction_v) {
        prediction = prediction_v.dup;
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        //writeln("serialize ErrorLayer");
        topology.topology ~= cast(size_t)LayerType.ErrorLayer;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        //writeln("deserialize ErrorLayer");
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ErrorLayer) {
            throw new Exception("Layer type mismatch");
        }
    }
}
class DenseL2 : ParameterLayer, Serializable {
    Parameters params;
    real[] input;
    real[] output;
    real[] nextGradient;
    size_t inputSize;
    size_t outputSize;
    real weightDecay; // L2 Regularization

    /**
     * Constructor initializes the weights, biases, and their gradients using the Parameters class.
     * Parameters:
     * - inputSize: The size of the input to the layer.
     * - outputSize: The size of the output from the layer.
     * - weightDecay: The weight decay factor for L2 regularization.
     */
    this(size_t inputSize, size_t outputSize, real weightDecay = 0.0) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weightDecay = weightDecay;
        params = new Parameters(inputSize, outputSize);
    }

    /**
     * Forward pass through the DenseL2 layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output.length = outputSize;
        foreach (i; 0 .. outputSize) {
            output[i] = params.biases[i];
            foreach (j; 0 .. inputSize) {
                output[i] += params.weights[i * inputSize + j] * input[j];
            }
        }
        return output;
    }

    /**
     * Backward pass through the DenseL2 layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        nextGradient.length = inputSize;
        nextGradient[] = 0.0;

        foreach (i; 0 .. outputSize) {
            params.biasGrads[i] += gradient[i];
            foreach (j; 0 .. inputSize) {
                size_t idx = i * inputSize + j;
                params.weightGrads[idx] += gradient[i] * input[j] + weightDecay * params.weights[idx];
                nextGradient[j] += params.weights[idx] * gradient[i];
            }
        }

        return nextGradient;
    }

    /**
     * Updates the weights and biases of the DenseL2 layer using the Parameters class.
     * Parameters:
     * - learningRate: The learning rate for the update.
     */
    override void update(real learningRate) {
        params.update(learningRate);
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.DenseL2;
        // Store the number of elements and dimensions
        topology.topology ~= inputSize;
        topology.topology ~= outputSize;
        // Store the weight decay in the params array
        topology.params ~= weightDecay;
        // Store the parameters
        params.serialize(topology);
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.DenseL2) {
            throw new Exception("Layer type mismatch");
        }

        // Retrieve the dimensions
        inputSize = topology.topology[1];
        outputSize = topology.topology[2];
        // Retrieve the weight decay from the params array
        weightDecay = topology.params[0];

        // Initialize the parameters
        params = new Parameters(inputSize, outputSize);
        // Retrieve the parameters
        size_t weightsStart = 1;
        size_t weightsEnd = weightsStart + inputSize * outputSize;
        size_t biasesStart = weightsEnd;
        size_t biasesEnd = biasesStart + outputSize;

        params.weights = topology.params[weightsStart .. weightsEnd];
        params.biases = topology.params[biasesStart .. biasesEnd];
    }
}

/**
 * Represents a ReLU (Rectified Linear Unit) activation layer.
 * Implements the Layer interface.
 */
class ActivationReLU : Layer, Serializable {
    real[] input;
    real[] output;

    /**
     * Forward pass through the ReLU activation layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output = new real[input.length];

        foreach (i, val; input) {
            output[i] = max(0.0, val);
        }

        return output;
    }

    /**
     * Backward pass through the ReLU activation layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        real[] reluDerivative = new real[input.length];
        foreach (i, val; input) {
            reluDerivative[i] = (val > 0.0) ? 1.0 : 0.0;
        }

        real[] nextGradient = new real[input.length];
        foreach (i, val; gradient) {
            nextGradient[i] = val * reluDerivative[i];
        }
        return nextGradient;
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.ActivationReLU;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ActivationReLU) {
            throw new Exception("Layer type mismatch");
        }
    }

}

/**
 * Represents a Leaky ReLU activation layer.
 * Implements the Layer interface.
 */
class ActivationLeakyReLU : Layer, Serializable {
    real[] input;
    real[] output;
    real alpha;

    /**
     * Constructor initializes the layer with a given alpha value.
     * Parameters:
     * - alpha: The negative slope for the Leaky ReLU function.
     */
    this(real alpha_in = 0.01) {
        alpha = alpha_in;
    }

    /**
     * Forward pass through the Leaky ReLU activation layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output = new real[input.length];

        foreach (i, val; input) {
            output[i] = max(alpha * val, val);
        }

        return output;
    }

    /**
     * Backward pass through the Leaky ReLU activation layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        real[] leakyReluDerivative = new real[input.length];
        foreach (i, val; input) {
            leakyReluDerivative[i] = (val > 0.0) ? 1.0 : alpha;
        }

        real[] nextGradient = new real[input.length];
        foreach (i, val; gradient) {
            nextGradient[i] = val * leakyReluDerivative[i];
        }
        return nextGradient;
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.ActivationLeakyReLU;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ActivationLeakyReLU) {
            throw new Exception("Layer type mismatch");
        }
    }

}

/**
 * Represents a Tanh (Hyperbolic Tangent) activation layer.
 * Implements the Layer interface.
 */
class ActivationTanh : Layer, Serializable {
    real[] input;
    real[] output;

    /**
     * Forward pass through the Tanh activation layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        input = input_v.dup;
        output = new real[input.length];

        foreach (i, val; input) {
            output[i] = tanh(val);
        }

        return output;
    }

    /**
     * Backward pass through the Tanh activation layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        real[] tanhDerivative = new real[input.length];
        foreach (i, val; output) {
            tanhDerivative[i] = 1.0 - pow(val, 2);
        }

        real[] nextGradient = new real[input.length];
        foreach (i, val; gradient) {
            nextGradient[i] = val * tanhDerivative[i];
        }
        return nextGradient;
    }

    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.ActivationTanh;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ActivationTanh) {
            throw new Exception("Layer type mismatch");
        }
    }
}


/**
 * Represents a Softmax activation layer.
 * Implements the Layer interface.
 */
class ActivationSoftmax : Layer, Serializable {
    real[] output;

    /**
     * Forward pass through the Softmax activation layer.
     * Parameters:
     * - input: The input data for the layer.
     * Returns:
     * - The output from the layer.
     */
    override real[] forward(real[] input_v) {
        real maxVal = input_v.maxElement;
        real[] expValues = input_v.map!(x => exp(x - maxVal)).array;
        real sumExp = expValues.sum;
        output = expValues.map!(x => x / sumExp).array;

        return output.dup;
    }

    /**
     * Backward pass through the Softmax activation layer.
     * Parameters:
     * - gradient: The gradient from the next layer.
     * Returns:
     * - The gradient for the current layer.
     */
    override real[] backward(real[] gradient) {
        // Ensure that output is correctly set from the forward pass
        if (output.length != gradient.length) {
            throw new Exception("Mismatch in dimensions between output and gradient");
        }

        real gradDotOutput = 0.0;
        foreach (g, o; zip(gradient, output)) {
            gradDotOutput += g * o;
        }

        real[] grad = new real[gradient.length];
        foreach (k; 0 .. gradient.length) {
            grad[k] = output[k] * (gradient[k] - gradDotOutput);
        }

        return grad.dup; // Return a copy of the gradient to avoid modifying the original data
    }


    override void serialize(NetworkTopology topology) {
        // Store the layer type
        topology.topology ~= cast(size_t)LayerType.ActivationSoftmax;
    }

    override void deserialize(NetworkTopology topology) {
        // Retrieve the layer type
        size_t layerType = topology.topology[0];
        if (layerType != cast(size_t)LayerType.ActivationSoftmax) {
            throw new Exception("Layer type mismatch");
        }
    }
}


/**
 * Computes the categorical cross-entropy loss for classification problems.
 * Parameters:
 * - trueLabels: The true labels in one-hot encoded form.
 * - probabilities: The predicted probabilities for each class.
 * Returns:
 * - The computed categorical cross-entropy loss.
 */
double categoricalCrossEntropyLoss(in real[] trueLabels, in real[] probabilities) {
    double loss = 0.0;
    foreach (i, prob; probabilities)
        if (trueLabels[i] != 0) // Assuming trueLabels is one-hot encoded
            loss -= trueLabels[i] * log(prob);
    return loss;
}



class Network {
    Layer[] layers;

    void addLayer(Layer layer) {
        layers ~= layer;
    }

    real[] forward(real[] input) {
        real[] output = input.dup;
        foreach (layer; layers) {
            output = layer.forward(output);
        }
        return output;
    }

    real[] backward(real[] gradient) {
        for (int i = cast(int) layers.length - 1; i >= 0; i--) {
            gradient = layers[i].backward(gradient);
        }
        return gradient;
    }

    void update(real learningRate) {
        foreach (layer; layers) {
            if (cast(ParameterLayer)layer !is null) {
                (cast(ParameterLayer)layer).update(learningRate);
            }
        }
    }

    NetworkTopology[] serialize() {
        NetworkTopology[] topologies = [];
        foreach (layer; layers) {
            NetworkTopology topology = new NetworkTopology;
            if (cast(Serializable)layer !is null) {
                (cast(Serializable)layer).serialize(topology);
            }
            topologies ~= topology;
        }
        //writeln(i"topologies length = $(topologies.length)");
        return topologies;
    }

    void deserialize(NetworkTopology[] topologies) {
        layers = [];
        //writeln(i"topologies length = $(topologies.length)");
        foreach (topology; topologies) {
            size_t layerType = topology.topology[0];
            Layer layer;
            //writeln(i"selecting layer $(layerType)");
            switch (layerType) {
                case cast(size_t)LayerType.Dense:
                    //writeln("Dense");
                    layer = new Dense(topology.topology[1], topology.topology[2]);
                    break;
                case cast(size_t)LayerType.ActivationSigmoid:
                    layer = new ActivationSigmoid();
                    break;
                case cast(size_t)LayerType.LayerNorm:
                    //writeln("LayerNorm");
                    layer = new LayerNorm(topology.topology[1]);
                    break;
                case cast(size_t)LayerType.DynamicTanh:
                    layer = new DynamicTanh(topology.topology[1]);
                    break;
                case cast(size_t)LayerType.ErrorLayer:
                    //writeln("ErrorLayer");
                    layer = new ErrorLayer();
                    break;
                case cast(size_t)LayerType.ActivationSoftmax:
                    layer = new ActivationSoftmax();
                    break;
                case cast(size_t)LayerType.DenseL2:
                    //writeln("DenseL2");
                    layer = new DenseL2(topology.topology[1], topology.topology[2], cast(real)topology.params[0]);
                    break;
                case cast(size_t)LayerType.ActivationReLU:
                    //writeln("ActivationReLU");
                    layer = new ActivationReLU();
                    break;
                case cast(size_t)LayerType.ActivationLeakyReLU:
                    layer = new ActivationLeakyReLU();
                    break;
                case cast(size_t)LayerType.ActivationTanh:
                    //writeln("ActivationTanh");
                    layer = new ActivationTanh();
                    break;
                default:
                    enforce(false, "Unknown layer type");
            }

            if (cast(Serializable)layer !is null) {
                (cast(Serializable)layer).deserialize(topology);
            }
            layers ~= layer;
        }
        //writeln("layer deserialize complete");
    }

    void serialize(string filename) {
        NetworkTopology[] topologies = serialize();
        File file;
        file.open(filename, "wb");
        foreach (topology; topologies) {
            topology.serialize(file);
        }
        file.close;
    }

    void deserialize(string filename) {
        File file;
        file.open(filename, "rb");
        NetworkTopology[] topologies;

        size_t pos = file.tell();
        size_t temp;

        while (!file.eof) {
            file.seek(pos);
            //writeln("reading layer");
            NetworkTopology topology = new NetworkTopology;
            //writeln("deserialize");

            topology.deserialize(file);
            topologies ~= topology;

            pos = file.tell();
            file.rawRead([temp]);
        }
        deserialize(topologies);
    }
}


unittest {
    // Test LayerNorm
    auto ln = new LayerNorm(3); // Feature size 3
    real[] input = [1.0, 2.0, 3.0];
    auto output = ln.forward(input.dup);

    // Check mean and variance of output
    real mean_out = output.sum / 3.0;
    real var_out = output.map!(x => pow(x - mean_out, 2)).sum / 3.0;
    assert(mean_out.approxEqual(0.0, 1e-6), "Mean not near zero");

    real original_var = input.map!(x => pow(x - input.sum/3, 2)).sum /3;
    real expected_var = original_var / (original_var + ln.epsilon);
    assert(var_out.approxEqual(expected_var, 1e-6), "Variance not as expected");

    // Test backward
    real[] grad = [1.0, 1.0, 1.0];
    auto grad_in = ln.backward(grad);

    // Check beta gradients
    real[] expected_beta = [1.0, 1.0, 1.0];
    assert(ln.betaGrads[].equal(expected_beta), "Beta gradients mismatch");

    // Check gamma gradients
    real[] expected_gamma = new real[3];
    foreach (i, g; grad) {
        expected_gamma[i] = g * ln.normalized[i];
    }
    foreach (i; 0..3) {
        assert(ln.gammaGrads[i].approxEqual(expected_gamma[i], 1e-6),
               "Gamma gradient at " ~ to!string(i) ~ " mismatch");
    }

    // Check input gradient (should be zero)
    real[] expected_dx = [0.0, 0.0, 0.0];
    assert(grad_in[].equal(expected_dx), "Input gradient mismatch");

    // Update parameters and check
    ln.update(0.1);
    real variance_input = (input.map!(x => pow(x - input.sum/3, 2)).sum)/3;
    real std_val = sqrt(variance_input + ln.epsilon);
    real inv_std = 1/std_val;

    // Check gamma[0]
    real gamma0_expected = 1.0 + 0.1 * (1.0 * inv_std);
    assert(ln.gamma[0].approxEqual(gamma0_expected, 1e-6), "Gamma0 update failed");

    // Check gamma[1]
    assert(ln.gamma[1].approxEqual(1.0, 1e-6), "Gamma1 update incorrect");

    // Check gamma[2]
    real gamma2_expected = 1.0 - 0.1 * inv_std;
    assert(ln.gamma[2].approxEqual(gamma2_expected, 1e-6), "Gamma2 update failed");

    // Check beta updates
    foreach (b; ln.beta) {
        assert(b.approxEqual(-0.1, 1e-6), "Beta update failed");
    }
}


unittest {
    writeln("neural network with LayerNorm and ErrorLayer -1");
    auto net = new Network();
    net.addLayer(new Dense(2, 7));
    net.addLayer(new LayerNorm(7));
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(7, 1));
    net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.1;
    size_t epochs = 3000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable idx; 0..inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally

            // Update parameters after processing all inputs in one epoch
            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f",
                   inputs[i], expected, prediction[0]));
    }
}


unittest {
    writeln("neural network with LayerNorm and ErrorLayer -2");
    auto net = new Network();
    net.addLayer(new Dense(2, 5));
    //net.addLayer(new LayerNorm(5));
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new DenseL2(5, 1));
    //net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.1;
    size_t epochs = 3000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable idx; 0..inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally
        }
        // Update parameters after processing all inputs in one epoch
        net.update(learningRate);
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f",
                   inputs[i], expected, prediction[0]));
    }
}

unittest {
    writeln("Testing DynamicTanh layer");

    size_t featureSize = 3;
    auto dytLayer = new DynamicTanh(featureSize);

    real[] input = [1.0, -2.0, 3.0];
    real[] output = dytLayer.forward(input);
    writeln("Forward output: ", output);

    real[] gradient = [0.5, -0.5, 0.5];
    real[] inputGradient = dytLayer.backward(gradient);
    writeln("Backward input gradient: ", inputGradient);

    // Update parameters with learning rate
    real learningRate = 0.01;
    dytLayer.update(learningRate);
}

unittest {
    writeln("neural network with DynamicTanh, ReLu -1");
version(unused)
{
    auto net = new Network();
    net.addLayer(new Dense(2, 3));
    net.addLayer(new DynamicTanh(3));
    net.addLayer(new ActivationReLU());
    //net.addLayer(new ActivationTanh());
    //net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(3, 1));
    //net.addLayer(new ActivationReLU());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);
}
    auto net = new Network();
    net.addLayer(new Dense(2, 5));
    net.addLayer(new LayerNorm(5));
    //net.addLayer(new ActivationTanh());
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.1;
    size_t epochs = 3000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0 .. epochs) {
        foreach (immutable idx; 0 .. inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally
        }
        // Update parameters after processing all inputs in one epoch

        net.update(learningRate);
        // Debug print every 100 epochs
        //if (ep % 100 == 0) {
        //    writeln("Epoch: ", ep);
        //    foreach (immutable i; 0 .. inputs.length) {
        //        real[] prediction = net.forward(inputs[i]);
        //        writeln("Input: ", inputs[i], ", Expected: ", targets[i], ", Prediction: ", prediction);
        //    }
        //}
    }

    foreach (immutable i; 0 .. inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f",
                   inputs[i], expected, prediction[0]));
    }
}

unittest {
    writeln("neural network with DynamicTanh -1");
    auto net = new Network();
    net.addLayer(new Dense(2, 5));
    net.addLayer(new DynamicTanh(5));
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.15;
    size_t epochs = 5000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0 .. epochs) {
        foreach (immutable idx; 0 .. inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally
        }
        // Update parameters after processing all inputs in one epoch
        net.update(learningRate);

        // Debug print every 100 epochs
        //if (ep % 100 == 0) {
        //    writeln("Epoch: ", ep);
        //    foreach (immutable i; 0 .. inputs.length) {
        //       real[] prediction = net.forward(inputs[i]);
        //        writeln("Input: ", inputs[i], ", Expected: ", targets[i], ", Prediction: ", prediction);
        // }
        //}
    }

    foreach (immutable i; 0 .. inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f",
                   inputs[i], expected, prediction[0]));
    }
}

unittest {
    writeln("neural network with DynamicTanh -2");
    auto net = new Network();
    net.addLayer(new Dense(2, 5));
    net.addLayer(new DynamicTanh(5)); // Replace LayerNorm with DynamicTanh
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.12;
    size_t epochs = 5000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0 .. epochs) {
        foreach (immutable idx; 0 .. inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally
        }
        // Update parameters after processing all inputs in one epoch
        net.update(learningRate);
    }

    foreach (immutable i; 0 .. inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f",
                   inputs[i], expected, prediction[0]));
    }
}

unittest {
    writeln("neural network with no output sigmoid and ErrorLayer");
    auto net = new Network();
    net.addLayer(new Dense(2, 7));          // First hidden layer
    net.addLayer(new ActivationSigmoid());  // Activation for hidden layer
    net.addLayer(new Dense(7, 1));          // Output layer (now direct without sigmoid)
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.12;
    size_t epochs = 2000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable bw; 0..inputs.length) {
            real[] output = net.forward(inputs[bw]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[bw]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally

            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f",
            inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.2f",
                inputs[i], expected, prediction[0]));
    }
}


unittest {
    writeln("neural network with output sigmoid and ErrorLayer");
    auto net = new Network();
    net.addLayer(new Dense(2, 6));
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(6, 1));
    net.addLayer(new ActivationSigmoid());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.8;
    size_t epochs = 1800;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable bw; 0..inputs.length) {
            real[] output = net.forward(inputs[bw]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[bw]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally

            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f", inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1,
            format("Input: %s, Expected: %.2f, Actual: %.4f", inputs[i], expected, prediction[0]));
    }
}

unittest {
    // Test Softmax layer
    auto softmax = new ActivationSoftmax();
    real[] input = [2.0, 1.0, 0.0];
    real[] output = softmax.forward(input);

    // Expected values (precomputed for verification)
    real expected0 = 0.6652409558;
    real expected1 = 0.2447284711;
    real expected2 = 0.0900305732;
    assert(approxEqual(output[0], expected0, 1e-6), "Forward 0 failed");
    writeln(i"expected1 = $(expected1)", format("%12.10f", output[1]));
    assert(approxEqual(output[1], expected1, 1e-6), "Forward 1 failed");
    writeln(i"expected2 = $(expected2)", format("%12.10f", output[2]));
    assert(approxEqual(output[2], expected2, 1e-6), "Forward 2 failed");

    real[] grad_in = [0.1, 0.2, 0.3];
    real[] grad_out = softmax.backward(grad_in);

    // Compute expected gradients (precomputed for verification)
    real exp_grad0 = -0.02825;
    real exp_grad1 = 0.01409;
    real exp_grad2 = 0.01418;
    assert(approxEqual(grad_out[0], exp_grad0, 1e-4), "Backward 0 mismatch");
    assert(approxEqual(grad_out[1], exp_grad1, 1e-4), "Backward 1 mismatch");
    assert(approxEqual(grad_out[2], exp_grad2, 1e-4), "Backward 2 mismatch");
}


unittest {
    writeln("Testing network with Softmax for XOR");
    auto net = new Network();

    // Configure network layers
    net.addLayer(new Dense(2, 3));
    net.addLayer(new ActivationSigmoid());
    net.addLayer(new Dense(3, 2));
    net.addLayer(new ActivationSoftmax());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.84;
    size_t epochs = 12000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ];

    foreach (ep; 0..epochs) {
        foreach (idx; 0..4) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            // Backward propagation
            net.backward([0.0, 0.0]); // Initial gradient for two outputs
            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real[] expected = targets[i];
        real diff0 = abs(prediction[0] - expected[0]);
        real diff1 = abs(prediction[1] - expected[1]);
        writeln(i"Input $(inputs[i]): Expected $(expected), got $(prediction)");
        assert(diff0 < 0.1 && diff1 < 0.1,
            format("Input %s: Expected %s, got %s", inputs[i], expected, prediction));
    }
}


unittest {
    writeln("neural network with Dense Layer, ReLU, and Leaky ReLU");
    auto net = new Network();
    net.addLayer(new DenseL2(2, 5, 0.001)); // Dense layer with L2 regularization
    net.addLayer(new ActivationReLU());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationLeakyReLU());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.07;
    size_t epochs = 2000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable idx; 0..inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally

            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f", inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1, format("Input: %s, Expected: %.2f, Actual: %.4f", inputs[i], expected, prediction[0]));
    }
}


unittest {
    writeln("neural network with DenseL2 Layer, ReLU, Leaky, and Tanh");
    auto net = new Network();
    net.addLayer(new DenseL2(2, 5, 0.001)); // Dense layer with L2 regularization
    net.addLayer(new ActivationReLU());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationTanh());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    real learningRate = 0.07;
    size_t epochs = 2000;
    real[][] inputs = [[0.0, 0.0], [0.0, 1.0], [1.4, 0.0], [1.0, 1.0]];
    real[][] targets = [[0.0], [1.0], [1.0], [0.0]];

    foreach (ep; 0..epochs) {
        foreach (immutable idx; 0..inputs.length) {
            real[] output = net.forward(inputs[idx]);
            errorLayer.setPrediction(output);
            errorLayer.forward(targets[idx]);

            net.backward([0.0]); // The ErrorLayer will compute the gradient internally

            net.update(learningRate);
        }
    }

    foreach (immutable i; 0..inputs.length) {
        real[] prediction = net.forward(inputs[i]);
        real expected = targets[i][0];
        writeln(format("Input: %s, Expected: %.2f, Actual: %.2f", inputs[i], expected, prediction[0]));
        assert(abs(prediction[0] - expected) < 0.1, format("Input: %s, Expected: %.2f, Actual: %.4f", inputs[i], expected, prediction[0]));
    }
}

unittest {
    writeln("Testing Serialization and Deserialization of Network");

    auto net = new Network();
    net.addLayer(new DenseL2(2, 5, 0.001)); // Dense layer with L2 regularization
    net.addLayer(new ActivationReLU());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new LayerNorm(5));
    net.addLayer(new ActivationTanh());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    writeln("serialize network");
    NetworkTopology[] topologies = net.serialize();

    auto deserializedNet = new Network();
    writeln("deserialize network");
    deserializedNet.deserialize(topologies);

    // Check if the deserialized network has the correct number of layers
    assert(deserializedNet.layers.length == net.layers.length, "Layer count mismatch");

    // Check if each layer type is the same
    foreach (i, layer; deserializedNet.layers) {
        auto expectedType = cast(LayerType) topologies[i].topology[0];
        writeln(i"expected type $(expectedType)");

        LayerType actualType;
        if (cast(ParameterLayer)layer !is null) {
            if (cast(Dense)layer !is null) {
                actualType = LayerType.Dense;
            }
            else if (cast(DenseL2)layer !is null) {
                actualType = LayerType.DenseL2;
            }
            else if (cast(LayerNorm)layer !is null) {
                actualType = LayerType.LayerNorm;
            }
        }
        else
        {
            if (cast(ActivationReLU)layer !is null) {
                actualType = LayerType.ActivationReLU;
            }
            else if (cast(ActivationTanh)layer !is null) {
                actualType = LayerType.ActivationTanh;
            }
            else if (cast(ActivationSoftmax)layer !is null) {
                actualType = LayerType.ActivationSoftmax;
            }
            else if (cast(ActivationSigmoid)layer !is null) {
                actualType = LayerType.ActivationSigmoid;
            }
            else if (cast(ActivationLeakyReLU)layer !is null) {
                actualType = LayerType.ActivationLeakyReLU;
            }
            else if (cast(ErrorLayer)layer !is null) {
                actualType = LayerType.ErrorLayer;
            }
        }
        // string message = i"$(actualType) != $(expectedType)"
        assert(actualType == expectedType, "Layer type mismatch at index " ~ to!string(i) ~ " " ~ to!string(actualType) );
    }

    writeln("layer type check complete");

    // Further checks can be done to ensure weights, biases, etc., are correctly serialized/deserialized
    // For example, for the first dense layer:
    auto firstLayer = cast(DenseL2)net.layers[0];
    auto firstDeserializedLayer = cast(DenseL2)deserializedNet.layers[0];
    assert(firstLayer.params.weights.equal(firstDeserializedLayer.params.weights), "Weights mismatch in first Dense layer");
    assert(firstLayer.params.biases.equal(firstDeserializedLayer.params.biases), "Biases mismatch in first Dense layer");

    // Check the second Dense layer
    auto secondLayer = cast(Dense)net.layers[2];
    auto secondDeserializedLayer = cast(Dense)deserializedNet.layers[2];
    assert(secondLayer.params.weights.equal(secondDeserializedLayer.params.weights), "Weights mismatch in second Dense layer");
    assert(secondLayer.params.biases.equal(secondDeserializedLayer.params.biases), "Biases mismatch in second Dense layer");
}

unittest {
    writeln("Testing Serialization and Deserialization of Network");

    auto net = new Network();
    net.addLayer(new DenseL2(2, 5, 0.001)); // Dense layer with L2 regularization
    net.addLayer(new DynamicTanh(5)); // Replace LayerNorm with DynamicTanh
    net.addLayer(new ActivationReLU());
    net.addLayer(new Dense(5, 1));
    net.addLayer(new ActivationTanh());
    auto errorLayer = new ErrorLayer();
    net.addLayer(errorLayer);

    writeln("serialize to file");
    // Serialize the network to a binary file
    string filename = "network_topology.bin";
    net.serialize(filename);

    writeln("deserialize from file");
    // Deserialize the network from the binary file
    auto deserializedNet = new Network();
    deserializedNet.deserialize(filename);

    writeln("check network");
    // Check if the deserialized network has the correct number of layers
    assert(deserializedNet.layers.length == net.layers.length, "Layer count mismatch");

version (unused)
{
    // Check if each layer type is the same
    foreach (i, layer; deserializedNet.layers) {
        auto expectedType = cast(LayerType) topology.topology[i];
        LayerType actualType;
        if (cast(ParameterLayer)layer !is null) {
            if (cast(Dense)layer !is null) {
                actualType = LayerType.Dense;
            }
            else if (cast(DenseL2)layer !is null) {
                actualType = LayerType.DenseL2;
            }
        }
        else if (cast(ActivationLayer)layer !is null) {
            if (cast(ActivationReLU)layer !is null) {
                actualType = LayerType.ActivationReLU;
            }
            else if (cast(ActivationTanh)layer !is null) {
                actualType = LayerType.ActivationTanh;
            }
            else if (cast(ActivationSoftmax)layer !is null) {
                actualType = LayerType.ActivationSoftmax;
            }
            else if (cast(ActivationSigmoid)layer !is null) {
                actualType = LayerType.ActivationSigmoid;
            }
            else if (cast(ActivationLeakyReLU)layer !is null) {
                actualType = LayerType.ActivationLeakyReLU;
            }
        }
        else if (cast(ErrorLayer)layer !is null) {
            actualType = LayerType.ErrorLayer;
        }
        assert(actualType == expectedType, "Layer type mismatch at index " ~ to!string(i));
    }
}

    // Further checks can be done to ensure weights, biases, etc., are correctly serialized/deserialized
    // For example, for the first dense layer:
    writeln("check first layer");
    auto firstLayer = cast(DenseL2)net.layers[0];
    auto firstDeserializedLayer = cast(DenseL2)deserializedNet.layers[0];
    assert(firstLayer.params.weights.equal(firstDeserializedLayer.params.weights), "Weights mismatch in first Dense layer");
    assert(firstLayer.params.biases.equal(firstDeserializedLayer.params.biases), "Biases mismatch in first Dense layer");

    // Check the second DynamicTanh layer
    auto secondLayer = cast(DynamicTanh)net.layers[1];
    auto secondDeserializedLayer = cast(DynamicTanh)deserializedNet.layers[1];
    assert(secondLayer.gamma.equal(secondDeserializedLayer.gamma), "Gamma mismatch in second DynamicTanh layer");
    assert(secondLayer.beta.equal(secondDeserializedLayer.beta), "Beta mismatch in second DynamicTanh layer");

    auto thirdLayer = cast(Dense)net.layers[3];
    auto thirdDeserializedLayer = cast(Dense)deserializedNet.layers[3];
    assert(thirdLayer.params.weights.equal(thirdDeserializedLayer.params.weights), "Weights mismatch in third DenseL2 layer");
    assert(thirdLayer.params.biases.equal(thirdDeserializedLayer.params.biases), "Biases mismatch in third DenseL2 layer");

}
