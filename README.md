#===============================================================
# SCRIPT COMPLETO - REDE NEURAL ARTIFICIAL (Deep Learning em R)
#===============================================================

# PARTE 0 - INSTALAÇÃO E REPRODUTIBILIDADE
# ---------------------------------------------------------------

# Instale os pacotes necessários (se ainda não os tiver)
# install.packages("readxl")
# install.packages("sigmoid")

# Carregar os pacotes
library(readxl)
library(sigmoid)

# Incluir procedimento para garantir a reprodutibilidade.
set.seed(0)

# PARTE 1 - PRÉ-PROCESSAMENTO DE DADOS
# ---------------------------------------------------------------

# Carregar os dados (ATENÇÃO: verifique e ajuste o caminho do arquivo)
# Mantenha o arquivo 'Data_TCC_Class_F.xlsx' neste caminho ou ajuste-o.
data <- read_excel("C:/users/Usuário/Desktop/RNA_Densa/Data_TCC_Class_F.xlsx")

# Definir conjunto de treino e teste (75% treino, 25% teste)
train_test_split_index <- 0.75 * nrow(data)

# Seu conjunto de dados tem 40 observações: 40 * 0.75 = 30 linhas para treino
train <- data.frame(data[1:train_test_split_index,])
test <- data.frame(data[(train_test_split_index + 1): nrow(data),])

# Definir as variáveis explicativas (X) e a variável target (Y)
train_x <- data.frame(train[1:3601])
train_y <- data.frame(train[3602])

test_x <- data.frame(test[1:3601])
test_y <- data.frame(test[3602])

# Transposição da matriz (Observações nas colunas, Variáveis nas linhas)
train_x <- t(train_x)
train_y <- t(train_y)

test_x <- t(test_x)
test_y <- t(test_y)


# PARTE 2 - FUNÇÕES DA REDE NEURAL (GENERALIZADAS)
# ---------------------------------------------------------------

# Função 1 - Criar a arquitetura da rede (AGORA SUPORTA MÚLTIPLAS CAMADAS)
getLayerSize <- function(X, y, hidden_layer_sizes) {
  n_x <- dim(X)[1] # Camada de entrada (3601)
  n_y <- dim(y)[1] # Camada de saída (1)
  
  # O vetor de tamanhos incluirá entrada, camadas escondidas e saída
  layer_sizes <- c(n_x, hidden_layer_sizes, n_y)
  
  size <- list("layer_sizes" = layer_sizes,
               "num_layers" = length(layer_sizes) - 1) # Número de pares (W, b)
  
  return(size)
}


# Função 2 - Inicializa Parâmetros randomicamente (AGORA SUPORTA MÚLTIPLAS CAMADAS)
initializeParameters <- function(layer_size){
  
  layer_sizes <- layer_size$layer_sizes
  num_layers <- layer_size$num_layers
  params <- list()
  
  # Loop para inicializar pesos (W) e bias (b) para todas as camadas
  for (l in 1:num_layers) {
    W_name <- paste0("W", l)
    b_name <- paste0("b", l)
    
    # Inicialização He (apenas para garantir que a rede mais profunda funcione melhor)
    # W[l]: matriz de layer_sizes[l+1] x layer_sizes[l]
    params[[W_name]] <- matrix(runif(layer_sizes[l+1] * layer_sizes[l], min = -1, max = 1), 
                               nrow = layer_sizes[l+1], ncol = layer_sizes[l]) * sqrt(2 / layer_sizes[l])
    
    # b[l]: vetor de layer_sizes[l+1] x 1
    params[[b_name]] <- matrix(0, nrow = layer_sizes[l+1], ncol = 1)
  }
  
  return (params)
}

# Funções de Ativação (Mantemos a Sigmoide conforme seu projeto original)
# A função sigmoide será chamada via 'sigmoid::sigmoid(Z)' no Forward Propagation.


# Função 3 - Forward Propagation (CORRIGIDA)
forwardPropagation <- function(X, params, layer_size){
  
  num_layers <- layer_size$num_layers
  A_prev <- X
  cache <- list("A0" = X) 
  
  # Loop de Propagação
  for (l in 1:num_layers) {
    W <- params[[paste0("W", l)]]
    b <- params[[paste0("b", l)]]
    
    # Z = W * A_prev + b 
    # CORREÇÃO APLICADA AQUI: Usando sweep() para somar o bias 'b' a cada coluna de W %*% A_prev
    Z_temp <- W %*% A_prev
    Z <- sweep(Z_temp, 1, b, "+") # Garante que 'b' (margem 1=linha) seja somado a cada coluna de Z_temp
    
    A <- sigmoid::sigmoid(Z)
    
    # Armazena Z e A no cache
    cache[[paste0("Z", l)]] <- Z
    cache[[paste0("A", l)]] <- A
    A_prev <- A 
  }
  
  cache[["A_final"]] <- A_prev 
  return (cache)
}


# Função 4 - Cost Function (Mean Squared Error)
computeCost <- function(y, cache) {
  m <- dim(y)[2]
  A_final <- cache[["A_final"]]
  
  # Custo (MSE - Mean Squared Error)
  cost <- sum((y - A_final)^2) / m
  
  return (cost)
}


# Função 5 - Backpropagation (AGORA SUPORTA MÚLTIPLAS CAMADAS)
backwardPropagation <- function(X, y, cache, params, layer_size){
  
  m <- dim(X)[2]
  num_layers <- layer_size$num_layers
  grads <- list()
  
  # Etapa 1: Calcular o erro (dZ) na CAMADA DE SAÍDA (L=num_layers)
  L <- num_layers
  A_final <- cache[[paste0("A", L)]]
  dZ <- A_final - y # dZ[L] (Erro do MSE)
  
  # Gradiente de W[L] e b[L]
  A_prev <- cache[[paste0("A", L - 1)]]
  grads[[paste0("dW", L)]] <- 1/m * (dZ %*% t(A_prev))
  grads[[paste0("db", L)]] <- 1/m * rowSums(dZ)
  
  # Etapa 2: Propagar o erro para trás (l = L-1 até 1)
  for (l in (L - 1):1) {
    W_next <- params[[paste0("W", l + 1)]]
    A_l <- cache[[paste0("A", l)]]
    A_prev <- cache[[paste0("A", l - 1)]]
    
    # Cálculo do dZ[l] (Propagação + Derivada da Sigmoide)
    # Derivada da Sigmoide: A * (1 - A)
    dZ <- (t(W_next) %*% dZ) * (A_l * (1 - A_l)) 
    
    # Gradientes de W[l] e b[l]
    grads[[paste0("dW", l)]] <- 1/m * (dZ %*% t(A_prev))
    grads[[paste0("db", l)]] <- 1/m * rowSums(dZ)
  }
  
  return(grads)
}


# Função 6 - Atualizar os pesos (AGORA SUPORTA MÚLTIPLAS CAMADAS)
updateParameters <- function(grads, params, learning_rate, layer_size){
  
  num_layers <- layer_size$num_layers
  updated_params <- params
  
  for (l in 1:num_layers) {
    W_name <- paste0("W", l)
    b_name <- paste0("b", l)
    dW_name <- paste0("dW", l)
    db_name <- paste0("db", l)
    
    # Atualização Gradiente Descendente
    updated_params[[W_name]] <- params[[W_name]] - learning_rate * grads[[dW_name]]
    updated_params[[b_name]] <- params[[b_name]] - learning_rate * grads[[db_name]]
  }
  
  return (updated_params)
}


# PARTE 3 - TREINAMENTO E AVALIAÇÃO DO MODELO
# ---------------------------------------------------------------

# Função 7 - Treinar o modelo (Loop Principal)
trainModel <- function(X, y, num_iteration, hidden_layers, lr){
  
  layer_size <- getLayerSize(X, y, hidden_layers)
  params <- initializeParameters(layer_size)
  
  cost_history <- c()
  
  for (i in 1:num_iteration) {
    # Forward
    fwd_prop <- forwardPropagation(X, params, layer_size)
    
    # Custo
    cost <- computeCost(y, fwd_prop)
    cost_history <- c(cost_history, cost)
    
    # Backward
    back_prop <- backwardPropagation(X, y, fwd_prop, params, layer_size)
    
    # Update
    params <- updateParameters(back_prop, params, learning_rate = lr, layer_size)
    
    # Opcional: Imprimir o custo a cada N iterações
    if (i %% 500 == 0) {
      cat(sprintf("Custo na iteração %d: %f\n", i, cost))
    }
  }
  
  model_out <- list("updated_params" = params,
                    "cost_hist" = cost_history)
  
  return (model_out)
}


# ------------------ HIPERPARÂMETROS ------------------
# Nova arquitetura com 3 camadas escondidas
HIDDEN_LAYERS = c(50, 20, 10) 
EPOCHS = 5000       # Aumentado para melhor convergência
LEARNING_RATE = 0.5 # Taxa de aprendizado ajustada (pode precisar de ajuste fino)
# -----------------------------------------------------

# Aplicar o treinamento
cat("\nIniciando o Treinamento do Modelo...\n")
train_model <- trainModel(train_x, train_y, 
                          hidden_layers = HIDDEN_LAYERS, 
                          num_iteration = EPOCHS, 
                          lr = LEARNING_RATE)
cat("Treinamento concluído.\n")

# PARTE 4 - RESULTADOS E AVALIAÇÃO
# ---------------------------------------------------------------

# 1. Avaliação Visual do Custo
plot(train_model$cost_hist, type = 'l', 
     main = "Histórico da Função Custo", 
     xlab = "Época", ylab = "Custo (MSE)")


# 2. Geração de Previsões no Conjunto de Teste
layer_size_test <- getLayerSize(test_x, test_y, HIDDEN_LAYERS)
params <- train_model$updated_params
fwd_prop_test <- forwardPropagation(test_x, params, layer_size_test)

# A previsão final (A_final)
y_pred <- fwd_prop_test$A_final

# Aplica um limiar (threshold) para classificação binária
# Valores acima de 0.5 são da classe 1, abaixo de 0.5 são da classe 0.
y_pred_class <- ifelse(y_pred > 0.5, 1, 0)


# 3. Comparação e Acurácia
cat("\nComparação entre Previsões (y_pred) e Real (test_y):\n")
compare <- rbind(Previsao = y_pred_class, Real = test_y)
print(compare)

accuracy <- mean(y_pred_class == test_y)
cat(sprintf("\nAcurácia no Conjunto de Teste: %.2f%%\n", accuracy * 100))
