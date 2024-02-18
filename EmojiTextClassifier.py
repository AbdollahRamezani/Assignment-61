import tensorflow as tf
import numpy as np
import time
import pandas as pd

class EmojiTextClassifier:
    def __init__(self) -> None:
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.word_vectors = None
        self.model = None

    def read_csv(self, file_path):
        df = pd.read_csv(file_path)
        X = np.array(df["sentence"])
        Y = np.array(df["label"], dtype=int)
        return X, Y

    def label_to_emoji(self, label):
        emojies = ["💚", "⚽️", "😍", "😞", "🍴"]
        return emojies[label]
    
    def load_dataset(self, dataset_path):
         self.X_train, self.Y_train = self.read_csv(f"{dataset_path}/train.csv")
         self.X_test, self.Y_test = self.read_csv(f"{dataset_path}/test.csv")

    def load_feature_vectors(self, file_path):
        self.word_vectors = {} 

        vector_file = open(file_path, encoding="utf-8")

        for line in vector_file:
            line = line.strip().split(" ")
            word = line[0]
            vector = np.array(line[1:], dtype = np.float64)
            self.word_vectors[word] = vector   

        return self.word_vectors 

    def sentence_to_feature_vectors_avg(self, sentence, word_vectors, dim):   
            try:
                sentence = sentence.lower()
                words = sentence.strip().split(" ")

                sum_vectors = np.zeros((dim, ))
                for word in words:
                    sum_vectors += word_vectors[word]

                self.avg_vector = sum_vectors / len(words)
                return self.avg_vector
            except:
                print(sentence)
                return None

    def load_model(self, dim):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, input_shape=(dim,), activation="softmax")
        ])    

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        ) 

    def train(self, word_vectors, dim):
        X_train_avg = []

        for x_train in self.X_train:
            X_train_avg.append(self.sentence_to_feature_vectors_avg(x_train, word_vectors, dim))

        X_train_avg = np.array(X_train_avg)     
        Y_train_one_hot = tf.keras.utils.to_categorical(self.Y_train, num_classes=5)    

        self.model.fit(X_train_avg, Y_train_one_hot, epochs=250)  


    def test(self, word_vectors, dim):
        X_test_avg = []

        for x_test in self.X_test:
            X_test_avg.append(self.sentence_to_feature_vectors_avg(x_test, word_vectors, dim))

        X_test_avg = np.array(X_test_avg)    
        Y_test_one_hot = tf.keras.utils.to_categorical(self.Y_test, num_classes=5)

        self.model.evaluate(X_test_avg, Y_test_one_hot)

    def predict(self, my_test, word_vectors, dim):
        my_test_avg = self.sentence_to_feature_vectors_ave(my_test, word_vectors, dim)  
        my_test_avg = np.array([my_test_avg])

        result = self.model.predict(my_test_avg)  
        y_pred = np.argmax(result)
        Emoji = self.label_to_emoji(y_pred)
        return Emoji
    

if __name__ == "main":


    etc = EmojiTextClassifier()

    dim = 300

    word_vectors = etc.load_feature_vectors("f/content/drive/MyDrive/Dataset/Emoji_Text_Classification/glove.6B/glove.6B.{dim}d.txt")    
    etc.load_dataset("/content/drive/MyDrive/Dataset/Emoji_Text_Classification")
    etc.load_model(dim)
    etc.train(word_vectors, dim)
    etc.test(word_vectors, dim)
    my_test = "I like eating in restaurants"
    etc.predict(my_test, word_vectors, dim)

    test_sentences, _ = etc.read_csv("/content/drive/MyDrive/Dataset/Emoji_Text_Classification/test.csv")
    n = len(test_sentences)

    start = time.time()

    for test_sentence in test_sentences:
        etc.predict(test_sentence, word_vectors, dim)

    inference_time = (time.time() - start) / n
    print("Inference time: ", inference_time)

