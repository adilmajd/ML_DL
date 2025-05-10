<h1 align="left">Master IT 2024-2025 en TAM</h1>

###
<h2 align="left">Projet 2 : Contrastive learning</h2>

###

<p align="left">Réalisé par :EL MAJDOUBI Adil & OUAHMANE Abdallah</p>
<p align="left">Encadé par : Pr .MAHMOUDI Abdelhak</p>


<h2 align="left">environment setup</h2>

###

<div align="left">
  <img src="https://avatars.githubusercontent.com/u/22800682?s=48&v=4" height="40" alt="javascript logo"  />
  <img width="12" />
  <img src="https://gravatar.com/avatar/5fcb1033abfa7fdb91f995d4035f6544" height="40" alt="typescript logo"  />
  <img width="12" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" height="40" alt="react logo"  />
  <img width="12" />

</div>

###

<h2 align="left">Installation les modules de python</h2>

```bash
pip install torch torchvision matplotlib
```


<h2 align="left">Documentation of the source code </h2>
###

```bash


"""
-- On utilise pytorch --
Le code met en œuvre un réseau siamois (Siamese Network) pour 
apprendre à distinguer si deux images de chiffres manuscrits (MNIST) 
représentent le même chiffre ou deux chiffres différents, 
en utilisant l'apprentissage contrastif.
"""




"""
* Import des modules pour les réseaux de neurones (torch, torchvision), les données, l'optimisation, etc.
* random pour créer des paires aléatoires d'images.
* matplotlib pour l'affichage.
"""
import torch
import torch.nn as nn #our construire le modèle
import torch.nn.functional as F #pour les fonctions d’activation
import torch.optim as optim #Pour l’optimiseur
from torchvision import datasets, transforms #datasets :Pour charger MNIST | transforms : Pour convertir les images en tenseurs .
from torch.utils.data import DataLoader, Dataset #Pour créer et gérer les données personnalisées.
import random #Pour créer des paires aléatoires d’images.
import matplotlib.pyplot as plt #Pour afficher les images à la fin.



"""
Création du Dataset personnalisé avec des paires:

* Crée des paires d'images : soit du même chiffre, soit de chiffres différents.
* La cible (label) est 1 si les chiffres sont identiques, 0 sinon.

Exemple :

    img1 = "2", img2 = "2" → label = 1

    img1 = "3", img2 = "7" → label = 0
"""
class ContrastiveMNIST(Dataset):
    #Le dataset MNIST standard est transformé pour générer des paires d'images.
    def __init__(self, mnist_dataset):
        self.data = mnist_dataset

    #img1 est choisie selon l’indice donné.
    #should_match détermine si on veut créer une paire semblable (1) ou différente (0).
    def __getitem__(self, index):
        img1, label1 = self.data[index]
        should_match = random.randint(0, 1)

        if should_match:
            #i la paire doit matcher, on boucle jusqu'à trouver une image ayant le même label que img1.
            # même classe
            while True:
                img2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 == label2:
                    break
        else:
            #Sinon, on cherche une image img2 avec un label différent.
            # classe différente
            while True:
                img2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 != label2:
                    break
        #On retourne deux images et un label binaire : 1 si même classe, 0 sinon.           
        return img1, img2, torch.tensor([int(label1 == label2)], dtype=torch.float32)

    def __len__(self):
        return len(self.data)






"""
Définition du réseau Siamese


* Deux images passent à travers le même CNN (forward_once).
* La distance entre leurs vecteurs est utilisée pour prédire si elles sont similaires.

Architecture :

    2 Convolutions + MaxPool

    2 couches fully connected

    Produit final : vecteur d'embedding de taille 128
"""
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #Partie convolutionnelle
        """
        Entrée : Image MNIST 1x28x28.

        2 blocs de convolution + activation + maxpool :

            Réduction progressive de la taille de l'image.

            Extraction de caractéristiques locales.
        """
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #Partie fully connected (dense)
        #Passage du résultat convolutif à un vecteur d’embedding de taille 128.
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    #Forward pour une seule image
    def forward_once(self, x):
        #On traite une image, on la met à plat, et on obtient un vecteur d’embedding.
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    #Forward pour une paire d’images
    def forward(self, x1, x2):
        #Applique le même réseau aux deux images.
        return self.forward_once(x1), self.forward_once(x2)







"""
Fonction de perte contrastive


* Si label = 1 (même classe), on minimise la distance.

* Si label = 0 (différentes), on maximise la distance jusqu'à une marge.

Formule :
    L=y⋅D²+(1-y)⋅max(0,m-D)²

où :

    - D est la distance euclidienne entre les deux embeddings

    - m est la marge (par défaut 1.0)
"""
#Marge : la distance minimale attendue entre deux classes différentes.
class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    #On calcule la distance euclidienne entre les deux embeddings.
    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)

        """
        Si label == 1 (même classe), on réduit la distance.
        Si label == 0, on augmente la distance jusqu'à margin.
        """
        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()



"""
Chargement et préparation des données


* Télécharge et transforme MNIST en tenseurs.
* Crée un dataset avec paires.
* Charge les données en batchs de taille 64.
"""
transform = transforms.Compose([transforms.ToTensor()])
train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = ContrastiveMNIST(train_mnist)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)




 

"""
Entraînement du modèle

À chaque itération :

    Passe img1 et img2 dans le modèle

    Calcule la perte avec ContrastiveLoss

    Applique backward() + optimizer.step()

    
"""

#Configuration : GPU si disponible.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
#Initialisation du modèle, de la fonction de perte et de l’optimiseur.
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Boucle d'entraînement
for epoch in range(5):
    total_loss = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #Affiche la perte totale par époque.
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")





"""
Visualisation

    Affiche les deux images et la distance entre leurs embeddings

    Calcule la distance de similarité.

    Affiche aussi un texte : "Same" ou "Different"
"""
def show_pair(img1, img2, distance, label):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1.squeeze(), cmap='gray')
    ax[1].imshow(img2.squeeze(), cmap='gray')
    plt.suptitle(f"Distance: {distance:.2f} - {'Same' if label else 'Different'}")
    plt.show()

test_img1, test_img2, test_label = train_dataset[0]
with torch.no_grad():
    e1, e2 = model(test_img1.unsqueeze(0).to(device), test_img2.unsqueeze(0).to(device))
    dist = F.pairwise_distance(e1, e2).item()
    show_pair(test_img1, test_img2, dist, test_label.item())


```
