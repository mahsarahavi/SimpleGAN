import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


file_path = "dataset.csv"
df = pd.read_csv(file_path)
df_numeric = df.drop(columns=['name', 'Target'])
df_target = df['Target']


def cluster_based_split(X, y, n_clusters=5, test_size=0.2, random_state=42):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X) 
    train_indices = []
    test_indices = []
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        if len(cluster_indices) > 1:
            train_index, test_index = train_test_split(cluster_indices, test_size=test_size, random_state=random_state)
            train_indices.extend(train_index)
            test_indices.extend(test_index)
        else:
            train_indices.extend(cluster_indices) 
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = cluster_based_split(df_numeric.values, df_target.values)


X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32).view(-1, 1)


loss_functions = {
    "MSELoss": nn.MSELoss(),
    "BCELoss": nn.BCELoss(),
}
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
}

data_size = X_train_tensor.shape[0]
input_size = X_train_tensor.shape[1]
noise_size = 10
hidden_size = 32
output_size = 1
epochs = 1000
lr = 0.001


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(noise_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, input_size + 1),
        nn.Sigmoid() 
    )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(input_size + 1, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid(),
    )
    def forward(self, x):
        return self.model(x)



def train_eval(loss_name, loss_fn, optimizer_name, optimizer_fn, noise_size, output_size):
    generator = Generator(noise_size)
    discriminator = Discriminator(input_size)

    optimizer_G = optimizer_fn(generator.parameters(), lr = lr)
    optimizer_D = optimizer_fn(discriminator.parameters(), lr = lr)

    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        discriminator.train()
        real_data = torch.cat((X_train_tensor, y_train_tensor), dim=1)
        fake_data = generator(torch.randn(real_data.size(0), noise_size))

        optimizer_D.zero_grad()

        real_labels = torch.ones(real_data.size(0), 1)
        fake_labels = torch.zeros(fake_data.size(0), 1)

        real_output = discriminator(real_data).view(-1, 1)
        fake_output = discriminator(fake_data.detach()).view(-1, 1)

        real_loss = loss_fn(real_output, real_labels)
        fake_loss = loss_fn(fake_output, fake_labels)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        generator.train()
        optimizer_G.zero_grad()

        g_loss = loss_fn(discriminator(fake_data), real_labels)
        g_loss.backward()
        optimizer_G.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        if epoch % 100 == 0:
            print(f"epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

            
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        noise_train = torch.randn(X_train_tensor.size(0), noise_size)
        noise_test = torch.randn(X_test_tensor.size(0), noise_size)
        generated_data_train = generator(noise_train)
        generated_data_test = generator(noise_test)
        predicted_y_train = generated_data_train[:, -1].view(-1, 1)
        predicted_y_test = generated_data_test[:, -1].view(-1, 1)


    return predicted_y_train, predicted_y_test, d_losses, g_losses



results = {}
best_g_loss = float('inf')
best_d_loss = float('inf')
best_generated_data = None
best_loss_name = None
best_optimizer = None


for loss_name, loss_fn in loss_functions.items():
    for optimizer_name, optimizer_fn in optimizers.items():
        print(f"Training with {loss_name} loss and {optimizer_name} optimizer...")
        
        predicted_y_train, predicted_y_test, d_losses, g_losses = train_eval(loss_name, loss_fn, optimizer_name, optimizer_fn, noise_size, output_size)
        results[(loss_name, optimizer_name)] = {
            'predicted_y_train' : predicted_y_train,
            'predicted_y_test' : predicted_y_test,
            'd_losses': d_losses,
            'g_losses': g_losses
        }
        final_d_loss = d_losses[-1]
        final_g_loss = g_losses[-1]

        if final_g_loss < best_g_loss:
            best_g_loss = final_g_loss
            best_d_loss = final_d_loss
            best_predicted_y_train = predicted_y_train
            best_predicted_y_test = predicted_y_test
            best_loss_name = loss_name
            best_optimizer = optimizer_name


print("Best training configuration:")
print(f"Loss Function: {best_loss_name}")
print(f"Optimizer: {best_optimizer}")
print(f"Best D Loss: {best_d_loss:.4f}")
print(f"Best G Loss: {best_g_loss:.4f}")


df_train = X_train.copy()
df_train = pd.DataFrame(df_train)
df_train["Actual_Target"] = y_train
df_train["Predicted_Target"] = best_predicted_y_train

df_test = X_test.copy()
df_test = pd.DataFrame(df_test)
df_test["Actual_Target"] = y_test
df_test["Predicted_Target"] = best_predicted_y_test

df_with_predictions = pd.concat([df_train, df_test])

df_with_predictions.to_csv("data_with_predictions_GAN.csv", index = False)

