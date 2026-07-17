#!/usr/bin/env python3
"""
Convert TrackPID_v1.dat format to ONNX model
Alternative version using TensorFlow/Keras which can export to ONNX

The .dat file contains neural network weights in text format:
  layer_name num_elements
  value1 value2 ... valueN
"""

import sys
import numpy as np
import argparse


def parse_dat_file(dat_filename):
    """
    Parse the .dat file and extract layer weights and biases
    Returns dict with layer names as keys and numpy arrays as values
    """
    layers = {}
    
    with open(dat_filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Parse layer header: "layer_name num_elements"
        parts = line.split()
        if len(parts) >= 2:
            layer_name = parts[0]
            try:
                num_elements = int(parts[1])
            except ValueError:
                i += 1
                continue
            
            # Read values from next line(s)
            values = []
            i += 1
            while len(values) < num_elements and i < len(lines):
                value_line = lines[i].strip()
                if value_line:
                    values.extend([float(x) for x in value_line.split()])
                i += 1
            
            # Store as numpy array
            if len(values) == num_elements:
                layers[layer_name] = np.array(values[:num_elements], dtype=np.float32)
                print(f"Loaded {layer_name}: shape {layers[layer_name].shape}")
            else:
                print(f"Warning: {layer_name} expected {num_elements} values but got {len(values)}")
        else:
            i += 1
    
    return layers


def create_onnx_model_tf(layers, output_file):
    """
    Create ONNX model using TensorFlow/Keras
    
    Network architecture (inferred from layer names):
    Input (4) -> Dense(5) -> Dense(10) -> Dense(5) -> Dense(1) -> Output
    """
    import tensorflow as tf
    from tensorflow import keras
    from tf2onnx.onnx_opset import onnxruntime_opset
    import onnx
    import tf2onnx.convert
    
    print("Using TensorFlow/Keras for model creation...")
    
    # Extract weights and biases for each layer
    w1 = layers.get('tensor_sequential1dense1CastReadVariableOp0', np.zeros((4, 5), dtype=np.float32))
    b1 = layers.get('tensor_sequential1dense1BiasAddReadVariableOp0', np.zeros(5, dtype=np.float32))
    
    w2 = layers.get('tensor_sequential1dense12CastReadVariableOp0', np.zeros((5, 10), dtype=np.float32))
    b2 = layers.get('tensor_sequential1dense12BiasAddReadVariableOp0', np.zeros(10, dtype=np.float32))
    
    w3 = layers.get('tensor_sequential1dense21CastReadVariableOp0', np.zeros((10, 5), dtype=np.float32))
    b3 = layers.get('tensor_sequential1dense21BiasAddReadVariableOp0', np.zeros(5, dtype=np.float32))
    
    w4 = layers.get('tensor_sequential1dense31CastReadVariableOp0', np.zeros((5, 1), dtype=np.float32))
    b4 = layers.get('tensor_sequential1dense31AddReadVariableOp0', np.array([0.0], dtype=np.float32))
    
    # Reshape weights to proper dimensions
    if w1.size == 20:
        w1 = w1.reshape(4, 5)
    if w2.size == 50:
        w2 = w2.reshape(5, 10)
    if w3.size == 50:
        w3 = w3.reshape(10, 5)
    if w4.size == 5:
        w4 = w4.reshape(5, 1)
    
    print(f"W1 shape: {w1.shape}, B1 shape: {b1.shape}")
    print(f"W2 shape: {w2.shape}, B2 shape: {b2.shape}")
    print(f"W3 shape: {w3.shape}, B3 shape: {b3.shape}")
    print(f"W4 shape: {w4.shape}, B4 shape: {b4.shape}")
    
    # Create Keras model
    model = keras.Sequential([
        keras.layers.Dense(5, activation='relu', input_shape=(4,), weights=[w1, b1]),
        keras.layers.Dense(10, activation='relu', weights=[w2, b2]),
        keras.layers.Dense(5, activation='relu', weights=[w3, b3]),
        keras.layers.Dense(1, weights=[w4, b4])
    ])
    
    model.summary()
    
    # Convert to ONNX
    print(f"Converting to ONNX and saving to {output_file}...")
    spec = (tf.TensorSpec((None, 4), tf.float32, name="input"),)
    output_path = output_file.replace('.onnx', '')
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    
    print(f"✓ Successfully converted to {output_file}")


def create_onnx_model_direct(layers, output_file):
    """
    Create ONNX model directly without external dependencies
    (using pure ONNX library if available)
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("Error: ONNX library not found. Install with: pip install onnx")
        return False
    
    print("Using pure ONNX for model creation...")
    
    # Extract weights and biases
    w1 = layers.get('tensor_sequential1dense1CastReadVariableOp0', np.zeros((4, 5), dtype=np.float32))
    b1 = layers.get('tensor_sequential1dense1BiasAddReadVariableOp0', np.zeros(5, dtype=np.float32))
    
    w2 = layers.get('tensor_sequential1dense12CastReadVariableOp0', np.zeros((5, 10), dtype=np.float32))
    b2 = layers.get('tensor_sequential1dense12BiasAddReadVariableOp0', np.zeros(10, dtype=np.float32))
    
    w3 = layers.get('tensor_sequential1dense21CastReadVariableOp0', np.zeros((10, 5), dtype=np.float32))
    b3 = layers.get('tensor_sequential1dense21BiasAddReadVariableOp0', np.zeros(5, dtype=np.float32))
    
    w4 = layers.get('tensor_sequential1dense31CastReadVariableOp0', np.zeros((5, 1), dtype=np.float32))
    b4 = layers.get('tensor_sequential1dense31AddReadVariableOp0', np.array([0.0], dtype=np.float32))
    
    # Reshape weights
    if w1.size == 20:
        w1 = w1.reshape(5, 4)
    if w2.size == 50:
        w2 = w2.reshape(10, 5)
    if w3.size == 50:
        w3 = w3.reshape(5, 10)
    if w4.size == 5:
        w4 = w4.reshape(1, 5)
    
    print(f"W1 shape: {w1.shape}, B1 shape: {b1.shape}")
    print(f"W2 shape: {w2.shape}, B2 shape: {b2.shape}")
    print(f"W3 shape: {w3.shape}, B3 shape: {b3.shape}")
    print(f"W4 shape: {w4.shape}, B4 shape: {b4.shape}")
    
    # Create initializers
    initializers = []
    
    initializers.append(helper.make_tensor(
        name="W1",
        data_type=TensorProto.FLOAT,
        dims=list(w1.shape),
        vals=w1.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        name="B1",
        data_type=TensorProto.FLOAT,
        dims=list(b1.shape),
        vals=b1.flatten().tolist()
    ))
    
    initializers.append(helper.make_tensor(
        name="W2",
        data_type=TensorProto.FLOAT,
        dims=list(w2.shape),
        vals=w2.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        name="B2",
        data_type=TensorProto.FLOAT,
        dims=list(b2.shape),
        vals=b2.flatten().tolist()
    ))
    
    initializers.append(helper.make_tensor(
        name="W3",
        data_type=TensorProto.FLOAT,
        dims=list(w3.shape),
        vals=w3.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        name="B3",
        data_type=TensorProto.FLOAT,
        dims=list(b3.shape),
        vals=b3.flatten().tolist()
    ))
    
    initializers.append(helper.make_tensor(
        name="W4",
        data_type=TensorProto.FLOAT,
        dims=list(w4.shape),
        vals=w4.flatten().tolist()
    ))
    initializers.append(helper.make_tensor(
        name="B4",
        data_type=TensorProto.FLOAT,
        dims=list(b4.shape),
        vals=b4.flatten().tolist()
    ))
    
    # Create input/output
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    
    # Create computation graph nodes
    nodes = [
        helper.make_node("Gemm", inputs=["input", "W1", "B1"], outputs=["Y1"], alpha=1.0, beta=1.0, transB=1),
        helper.make_node("Relu", inputs=["Y1"], outputs=["A1"]),
        helper.make_node("Gemm", inputs=["A1", "W2", "B2"], outputs=["Y2"], alpha=1.0, beta=1.0, transB=1),
        helper.make_node("Relu", inputs=["Y2"], outputs=["A2"]),
        helper.make_node("Gemm", inputs=["A2", "W3", "B3"], outputs=["Y3"], alpha=1.0, beta=1.0, transB=1),
        helper.make_node("Relu", inputs=["Y3"], outputs=["A3"]),
        helper.make_node("Gemm", inputs=["A3", "W4", "B4"], outputs=["output"], alpha=1.0, beta=1.0, transB=1),
    ]
    
    # Create graph and model
    graph = helper.make_graph(
        nodes=nodes,
        name="TrackPID_Network",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers
    )
    
    model = helper.make_model(graph, producer_name="Mu2e_TrackPID_Converter", opset_imports=[helper.make_opsetid("", 12)])
    
    # Save and verify
    print(f"Saving to {output_file}...")
    onnx.save(model, output_file)
    
    print("Verifying model...")
    onnx.checker.check_model(model)
    print("✓ Model verification passed!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert TrackPID_v1.dat to ONNX format")
    parser.add_argument("input", help="Input .dat file")
    parser.add_argument("-o", "--output", help="Output .onnx file", default="TrackPID_v1.onnx")
    parser.add_argument("--use-tf", action="store_true", help="Use TensorFlow/Keras for conversion (if available)")
    
    args = parser.parse_args()
    
    print(f"Reading {args.input}...")
    layers = parse_dat_file(args.input)
    
    print(f"\nLoaded {len(layers)} layer(s)")
    
    if args.use_tf:
        try:
            create_onnx_model_tf(layers, args.output)
        except ImportError as e:
            print(f"TensorFlow not available ({e}), falling back to direct ONNX creation...")
            create_onnx_model_direct(layers, args.output)
    else:
        success = create_onnx_model_direct(layers, args.output)
        if not success:
            print("\nTrying TensorFlow approach...")
            try:
                create_onnx_model_tf(layers, args.output)
            except Exception as e:
                print(f"TensorFlow approach failed: {e}")
                sys.exit(1)


if __name__ == "__main__":
    main()
