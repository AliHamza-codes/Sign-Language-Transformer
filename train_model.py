import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import TensorBoard

# 1. Data Loading Logic with Error Handling
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        try:
            # Check if all frames for this sequence exist
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            
            sequences.append(window)
            labels.append(label_map[action])
        except FileNotFoundError:
            # Agar koi frame missing hai to is poori video/sequence ko skip kar dein
            print(f"Skipping sequence {sequence} for action {action} due to missing files.")
            continue

# Data conversion
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Training aur Testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# 2. Optimized Transformer Architecture
def transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Self-Attention Block
    # key_dim ko input feature size ke mutabiq set kiya hai
    attention_output = MultiHeadAttention(num_heads=8, key_dim=input_shape[-1])(inputs, inputs)
    attention_output = Dropout(0.2)(attention_output)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward Block
    ff_output = Dense(256, activation="relu")(x)
    ff_output = Dense(input_shape[-1])(ff_output)
    x = LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    # Classification Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# 3. Model Initialization
model = transformer_model(X.shape[1:], actions.shape[0])

# Optimizer setup
model.compile(optimizer='Adam', 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

print(f"Data Loaded: {X.shape[0]} total sequences found.")
print("Training shuru ho rahi hai...")

# 4. Training
# Epochs thode zyada rakhe hain taake Transformer pattern samajh sake
model.fit(X_train, y_train, 
          epochs=150, 
          validation_data=(X_test, y_test), 
          batch_size=8)

# 5. Save Model
model.save('action.h5')
print("Model trained and saved as 'action.h5'!")