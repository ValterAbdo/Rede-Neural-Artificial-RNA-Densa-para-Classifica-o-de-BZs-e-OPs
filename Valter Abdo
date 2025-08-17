# Título: Aplicar RNA aos dados obtidos dos espectros teóricos das moléculas
# do tipo Opioides e Benzodiazepínicos para fins de classificação.
# Esta versão foi ajustada para a base de dados em formato .XLSX.

# Parte 1 - Incluir procedimento para garantir a reprodutibilidade.
# Garante que os resultados de geração de números aleatórios são os mesmos a cada execução.
set.seed(0)

# -------------------- Bibliotecas Necessárias --------------------
# O pacote 'readxl' é necessário para ler arquivos .xlsx.
# Instale-o se ainda não tiver: install.packages("readxl")
library(readxl)

# -------------------- Parte 2 - Baixar e Preparar os dados --------------------

# O nome do arquivo foi fornecido. Ajustando para o arquivo .XLSX.
file_name <- "Data_TCC_Class_F.xlsx - DF.csv" # O nome do arquivo que você forneceu
sheet_name <- "DF" # Presumindo que os dados estão na aba 'DF' do seu arquivo. Por favor, ajuste se necessário.

tryCatch({
  # Lendo o arquivo XLSX. O cabeçalho é lido por padrão.
  # Usamos read_excel da biblioteca readxl.
  full_data <- read_excel(file_name, sheet = sheet_name)
  
  if (nrow(full_data) == 0 || ncol(full_data) < 2) {
    stop("O arquivo de dados está vazio ou não contém dados suficientes. Verifique o nome do arquivo ou a aba.")
  }

  print(paste("1. Dimensões do dataframe original após a leitura:", nrow(full_data), "linhas x", ncol(full_data), "colunas"))

  # Separando as variáveis explicativas (X) e a variável explicada (Y).
  # As primeiras 3601 colunas são X e a última coluna (a 3602ª) é Y.
  X_raw <- full_data[, 1:3601]
  Y_raw <- full_data[, 3602]
  
  # Converter os dataframes para matrizes numéricas.
  X <- as.matrix(X_raw)
  Y <- as.matrix(Y_raw)

  print(paste("2. Dimensões da matriz X (samples x features) após a conversão:", nrow(X), "x", ncol(X)))
  print(paste("3. Dimensões da matriz Y (samples x 1) após a conversão:", nrow(Y), "x", ncol(Y)))

  num_samples <- nrow(X)
  
  # Normalizando os dados.
  normalizeData <- function(data) {
    min_val <- min(data)
    max_val <- max(data)
    if (max_val - min_val == 0) {
      return(matrix(0, nrow = nrow(data), ncol = ncol(data)))
    }
    normalized_data <- (data - min_val) / (max_val - min_val)
    return(normalized_data)
  }

  # Converter os rótulos para binário (0 ou 1).
  # Esta função depende da natureza dos seus dados Y. Se Y já for 0 ou 1, esta linha pode ser ajustada.
  Y <- ifelse(Y > median(Y), 1, 0)

  # Definir conjunto de treino e teste (Train Test Split).
  # Usando 75% dos dados para treino e 25% para teste.
  train_test_split_index <- floor(0.75 * num_samples)
  
  train_x_df <- X[1:train_test_split_index, , drop = FALSE]
  train_y_df <- Y[1:train_test_split_index, , drop = FALSE]
  
  test_x_df <- X[(train_test_split_index + 1):num_samples, , drop = FALSE]
  test_y_df <- Y[(train_test_split_index + 1):num_samples, , drop = FALSE]

  # Transpor as matrizes para o formato esperado pela rede neural (features x samples).
  # A normalização agora é aplicada a cada conjunto de dados transposto.
  train_x <- t(normalizeData(train_x_df))
  test_x <- t(normalizeData(test_x_df))
  train_y <- t(train_y_df)
  test_y <- t(test_y_df)
  
  print("4. Após transposição para o modelo de rede neural:")
  print(paste("Dimensões de train_x (features x samples):", nrow(train_x), "x", ncol(train_x)))
  print(paste("Dimensões de test_x (features x samples):", nrow(test_x), "x", ncol(test_x)))

}, error = function(e) {
  stop("Erro ao carregar ou processar o arquivo. Verifique se o nome do arquivo e o nome da aba estão corretos. Detalhes do erro: ", e$message)
})

# -------------------- Funções da Rede Neural --------------------

# Funções de ativação
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}
relu <- function(x) {
  return(pmax(0, x))
}

# Derivadas das funções de ativação
relu_deriv <- function(x) {
  return(ifelse(x > 0, 1, 0))
}

# Função 1 - Criar a arquitetura da rede.
# Agora com três camadas ocultas
getLayerSize <- function(X, y, hidden_neurons1, hidden_neurons2, hidden_neurons3) {
  # O número de features (n_x) é o número de linhas da matriz de entrada X (que está transposta).
  n_x <- dim(X)[1]
  n_h1 <- hidden_neurons1
  n_h2 <- hidden_neurons2
  n_h3 <- hidden_neurons3
  n_y <- dim(y)[1]
  
  size <- list(
    "n_x" = n_x,
    "n_h1" = n_h1,
    "n_h2" = n_h2,
    "n_h3" = n_h3,
    "n_y" = n_y
  )
  
  return(size)
}

# Função 2 - Inicializar parâmetros randomicamente.
# Agora para três camadas ocultas
initializeParameters <- function(layer_size) {
  n_x <- layer_size$n_x
  n_h1 <- layer_size$n_h1
  n_h2 <- layer_size$n_h2
  n_h3 <- layer_size$n_h3
  n_y <- layer_size$n_y
  
  W1 <- matrix(runif(n_h1 * n_x), nrow = n_h1, ncol = n_x, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, n_h1), nrow = n_h1, ncol = 1)
  
  W2 <- matrix(runif(n_h2 * n_h1), nrow = n_h2, ncol = n_h1, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, n_h2), nrow = n_h2, ncol = 1)
  
  W3 <- matrix(runif(n_h3 * n_h2), nrow = n_h3, ncol = n_h2, byrow = TRUE) * 0.01
  b3 <- matrix(rep(0, n_h3), nrow = n_h3, ncol = 1)
  
  W4 <- matrix(runif(n_y * n_h3), nrow = n_y, ncol = n_h3, byrow = TRUE) * 0.01
  b4 <- matrix(rep(0, n_y), nrow = n_y, ncol = 1)
  
  params <- list(
    "W1" = W1, "b1" = b1,
    "W2" = W2, "b2" = b2,
    "W3" = W3, "b3" = b3,
    "W4" = W4, "b4" = b4
  )
  
  # Adicionando a verificação de dimensão aqui.
  print(paste("Dimensões de b1:", nrow(params$b1), "x", ncol(params$b1)))
  
  return(params)
}

# Função 3 - Forward Propagation.
# Agora para três camadas ocultas
forwardPropagation <- function(X, params) {
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  W3 <- params$W3
  b3 <- params$b3
  W4 <- params$W4
  b4 <- params$b4
  
  print(paste("5. Dimensões de W1:", nrow(W1), "x", ncol(W1)))
  print(paste("6. Dimensões de X:", nrow(X), "x", ncol(X)))
  
  # Camada Oculta 1 - Usando sweep() para garantir que a soma seja compatível.
  Z1 <- sweep(W1 %*% X, 1, b1, "+")
  A1 <- relu(Z1)
  
  # Camada Oculta 2 - Usando sweep()
  Z2 <- sweep(W2 %*% A1, 1, b2, "+")
  A2 <- relu(Z2)
  
  # Camada Oculta 3 - Usando sweep()
  Z3 <- sweep(W3 %*% A2, 1, b3, "+")
  A3 <- relu(Z3)
  
  # Camada de Saída - Usando sweep()
  Z4 <- sweep(W4 %*% A3, 1, b4, "+")
  A4 <- sigmoid(Z4)
  
  cache <- list(
    "Z1" = Z1, "A1" = A1,
    "Z2" = Z2, "A2" = A2,
    "Z3" = Z3, "A3" = A3,
    "Z4" = Z4, "A4" = A4
  )
  
  return(cache)
}

# Função 4 - Cost Function com Regularização L2.
computeCost <- function(y, cache, params, lambda = 0.01) {
  m <- dim(y)[2]
  A4 <- cache$A4
  
  # Termo de Regularização L2
  W1 <- params$W1
  W2 <- params$W2
  W3 <- params$W3
  W4 <- params$W4
  l2_cost <- (lambda / (2 * m)) * (sum(W1^2) + sum(W2^2) + sum(W3^2) + sum(W4^2))
  
  # Custo do Erro Quadrático Médio
  cost <- sum((y - A4)^2) / m + l2_cost
  return(cost)
}

# Função 5 - Backpropagation.
backwardPropagation <- function(X, y, cache, params, lambda = 0.01) {
  m <- dim(X)[2]
  
  A1 <- cache$A1
  A2 <- cache$A2
  A3 <- cache$A3
  A4 <- cache$A4
  
  Z1 <- cache$Z1
  Z2 <- cache$Z2
  Z3 <- cache$Z3
  
  W2 <- params$W2
  W3 <- params$W3
  W4 <- params$W4
  
  # Saída (camada 4)
  dZ4 <- A4 - y
  dW4 <- (1/m) * (dZ4 %*% t(A3)) + (lambda/m) * W4
  db4 <- (1/m) * rowSums(dZ4)
  
  # Camada 3 (oculta)
  dZ3 <- (t(W4) %*% dZ4) * relu_deriv(Z3)
  dW3 <- (1/m) * (dZ3 %*% t(A2)) + (lambda/m) * W3
  db3 <- (1/m) * rowSums(dZ3)
  
  # Camada 2 (oculta)
  dZ2 <- (t(W3) %*% dZ3) * relu_deriv(Z2)
  dW2 <- (1/m) * (dZ2 %*% t(A1)) + (lambda/m) * W2
  db2 <- (1/m) * rowSums(dZ2)
  
  # Camada 1 (oculta)
  dZ1 <- (t(W2) %*% dZ2) * relu_deriv(Z1)
  dW1 <- (1/m) * (dZ1 %*% t(X)) + (lambda/m) * W1
  db1 <- (1/m) * rowSums(dZ1)
  
  grads <- list(
    "dW1" = dW1, "db1" = db1,
    "dW2" = dW2, "db2" = db2,
    "dW3" = dW3, "db3" = db3,
    "dW4" = dW4, "db4" = db4
  )
  
  return(grads)
}

# Função 6 - Atualizar os pesos.
updateParameters <- function(grads, params, learning_rate) {
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  W3 <- params$W3
  b3 <- params$b3
  W4 <- params$W4
  b4 <- params$b4
  
  dW1 <- grads$dW1
  db1 <- grads$db1
  dW2 <- grads$dW2
  db2 <- grads$db2
  dW3 <- grads$dW3
  db3 <- grads$db3
  dW4 <- grads$dW4
  db4 <- grads$db4
  
  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2
  W3 <- W3 - learning_rate * dW3
  b3 <- b3 - learning_rate * db3
  W4 <- W4 - learning_rate * dW4
  b4 <- b4 - learning_rate * db4
  
  updated_params <- list(
    "W1" = W1, "b1" = b1,
    "W2" = W2, "b2" = b2,
    "W3" = W3, "b3" = b3,
    "W4" = W4, "b4" = b4
  )
  
  return(updated_params)
}

# Função 7 - Função de predição
predict <- function(X, params) {
  fwd_prop <- forwardPropagation(as.matrix(X), params)
  y_pred_prob <- fwd_prop$A4
  y_pred_class <- ifelse(y_pred_prob > 0.5, 1, 0)
  return(y_pred_class)
}

# Função 8 - Calcular a acurácia do modelo
calculateAccuracy <- function(predictions, actual_labels) {
  correct_predictions <- sum(predictions == actual_labels)
  total_predictions <- length(predictions)
  accuracy <- (correct_predictions / total_predictions) * 100
  return(accuracy)
}

# -------------------- Parte 9 - Treinar modelo --------------------
trainModel <- function(X, y, num_iteration, hidden_neurons1, hidden_neurons2, hidden_neurons3, initial_lr, decay_rate, patience = 20, min_delta = 1e-5) {
  
  layer_size <- getLayerSize(X, y, hidden_neurons1, hidden_neurons2, hidden_neurons3)
  init_params <- initializeParameters(layer_size)
  
  cost_history <- c()
  best_cost <- Inf
  patience_counter <- 0
  
  for (i in 1:num_iteration) {
    current_lr <- initial_lr * (1 / (1 + decay_rate * i))
    
    fwd_prop <- forwardPropagation(X, init_params)
    cost <- computeCost(y, fwd_prop, init_params)
    
    if (i %% 100 == 0) {
      print(paste("Época", i, "- Custo:", cost))
    }
    
    if (i > 1 && abs(cost_history[length(cost_history)] - cost) < min_delta) {
        patience_counter <- patience_counter + 1
        if (patience_counter >= patience) {
            print(paste("Parando o treinamento na época", i, "devido a 'early stopping'."))
            break
        }
    } else {
        patience_counter <- 0
    }
    
    back_prop <- backwardPropagation(X, y, fwd_prop, init_params)
    update_params <- updateParameters(back_prop, init_params, learning_rate = current_lr)
    init_params <- update_params
    cost_history <- c(cost_history, cost)
  }
  
  model_out <- list(
    "updated_params" = update_params,
    "cost_hist" = cost_history
  )
  
  return(model_out)
}

# -------------------- Execução e Avaliação --------------------

# Definir hiperparâmetros
# Ajuste as camadas ocultas conforme seu pedido
HIDDEN_NEURONS1 = 40
HIDDEN_NEURONS2 = 20
HIDDEN_NEURONS3 = 10 
EPOCHS = 1000
LEARNING_RATE = 0.01
DECAY_RATE = 0.01

# Treinar o modelo
train_model <- trainModel(
  train_x,
  train_y,
  hidden_neurons1 = HIDDEN_NEURONS1,
  hidden_neurons2 = HIDDEN_NEURONS2,
  hidden_neurons3 = HIDDEN_NEURONS3,
  num_iteration = EPOCHS,
  initial_lr = LEARNING_RATE,
  decay_rate = DECAY_RATE
)

# Testar o modelo no conjunto de teste
params <- train_model$updated_params
y_pred <- predict(test_x, params)

print("Comparação de Previsões (y_pred) vs. Valores Reais (test_y):")
print(rbind(y_pred, test_y))

# Calcular e exibir a acurácia
accuracy <- calculateAccuracy(y_pred, test_y)
print(paste("Acurácia do modelo no conjunto de teste:", round(accuracy, 2), "%"))

# Plota o histórico de custo para verificar o progresso do treinamento.
plot(train_model$cost_hist, type = "l", xlab = "Época", ylab = "Custo", main = "Histórico de Custo durante o Treinamento")

