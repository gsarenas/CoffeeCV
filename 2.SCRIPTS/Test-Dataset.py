import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

def load_coffee_test():
    csv_path = "QSVM_TEST_SET_NOVO_DATASET.csv"
    return pd.read_csv(csv_path)

# Carregando o set de teste
test_set = load_coffee_test()

# Carregando o modelo treinado
svm_model = load('QSVM_TREINADO_NOVO_DATASET.joblib')

# Preparando os dados para teste
x_test = test_set[['R', 'G', 'B']]
y_test = test_set['Class']

# Efetuando as predicoes
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Acurácia: {}'.format(accuracy))
print(y_pred)