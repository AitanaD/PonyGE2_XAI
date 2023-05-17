
"""Variable que almacena el nombre del dataset que se esstá procesadno"""
dataset_name = ""

"""Variable que almacena la opción de ejecución seleccionada para el algoritmo G3P"""
opcion_ejec = ""

"""Variable que almacena el modelo de caja negra seleccionado para que 
    PonyGE entrene sobre él"""
modelo_bb = ""

"""Variable que almacena la iteración actual de los K-folds"""
iteracion = 0

"""Variable que almacena la lista de resultados f1 de entrenamiento en el
    modelo de caja negra usado"""
train_bb_f1 = list()

"""Variable que almacena la lista de resultados f1 de prueba en el
    modelo de caja negra usado"""
test_bb_f1 = list()

"""Variable que almacena la lista de resultados f1 de entrenamiento en el
    algorítmo PonyGE"""
train_pony_f1 = list()

"""Variable que almacena la lista de resultados f1 de test en el
    algoritmo PonuGE"""
test_pony_f1 = list()