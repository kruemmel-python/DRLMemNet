

# Experimenteller DRLMemNet

## Einführung

Dieses Projekt implementiert den Experimentellen DRLMemNet, ein neuronales Netzwerk mit differenzierbarem Speicher und einem Controller, der auf Deep Reinforcement Learning (DRL) basiert. Der Hauptzweck dieses Modells ist es, sequenzielle Daten effektiv zu verarbeiten und zu speichern, um die Leistung bei Aufgaben wie Textverarbeitung und Sprachmodellierung zu verbessern.

## Inhaltsverzeichnis

1. [Einführung](#einführung)
2. [Installation](#installation)
3. [Daten](#daten)
4. [Konfiguration](#konfiguration)
5. [Modellarchitektur](#modellarchitektur)
6. [Training](#training)
7. [Evaluierung](#evaluierung)
8. [Ergebnisse](#ergebnisse)
9. [Schlussfolgerung](#schlussfolgerung)
10. [Zukünftige Arbeit](#zukünftige-arbeit)
11. [Literatur](#literatur)

## Installation

Um das Projekt zu installieren und auszuführen, benötigen Sie Python 3.6 oder höher. Zusätzlich müssen die folgenden Bibliotheken installiert werden:

```bash
pip install torch numpy scikit-learn matplotlib tqdm pyyaml
```

## Daten

Die verwendeten Daten sind in der Datei `data/augmented_text_data.txt` gespeichert. Diese Datei enthält vorverarbeitete Textdaten, die für das Training und die Evaluierung des Modells verwendet werden.

## Konfiguration

Die Konfiguration des Modells erfolgt über die Datei `config.yaml`. Diese Datei enthält alle relevanten Parameter, die für das Training und die Evaluierung des Modells erforderlich sind.

```yaml
data_path: "data/augmented_text_data.txt"
log_file: "logs/training.log"
embedding_dim: 128
memory_size: 512
learning_rate: 0.01
batch_size: 16
max_seq_length: 100
train_size_ratio: 0.8
val_size_ratio: 0.1
epochs: 50
accumulation_steps: 1
write_strength: 0.1
patience: 3
save_path: "models/"
```

## Modellarchitektur

Das Modell besteht aus zwei Hauptkomponenten:

1. **Differentiable Memory**: Ein differenzierbarer Speicher, der es ermöglicht, Informationen über mehrere Zeitschritte zu speichern und abzurufen.
2. **Controller**: Ein neuronales Netzwerk, das die Eingaben verarbeitet und die Interaktion mit dem Speicher steuert.

### Differentiable Memory

Die DifferentiableMemory-Klasse implementiert einen differenzierbaren Speicher, der es ermöglicht, Schlüssel-Wert-Paare zu speichern und abzurufen. Die Klasse verwendet eine Aufmerksamkeitsmechanismus, um relevante Informationen aus dem Speicher abzurufen.

```python
class DifferentiableMemory(nn.Module):
    def __init__(self, memory_size: int, embedding_size: int, device=DEVICE):
        super(DifferentiableMemory, self).__init__()
        self.keys = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.values = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.device = device
        self._initialize_weights()
        self.usage_counts = nn.Parameter(torch.ones(memory_size, device=device), requires_grad=False)

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.keys)
        nn.init.kaiming_uniform_(self.values)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.keys.T)
        attention_weights = F.softmax(scores, dim=-1)
        memory_output = torch.matmul(attention_weights, self.values)
        activated_memory_indices = torch.argmax(scores, dim=-1)
        self.usage_counts[activated_memory_indices] += 1
        return memory_output

    def write(self, updated_keys: torch.Tensor, updated_values: torch.Tensor, write_strength: float):
        updated_keys = F.normalize(updated_keys, dim=-1)
        updated_values = F.normalize(updated_values, dim=-1)
        batch_size = updated_keys.shape[0]
        if batch_size != self.memory_size:
            updated_keys = updated_keys.mean(dim=0, keepdim=True).expand(self.memory_size, -1)
            updated_values = updated_values.mean(dim=0, keepdim=True).expand(self.memory_size, -1)
        with torch.no_grad():
            self.keys.copy_((1 - write_strength) * self.keys.data + write_strength * updated_keys)
            self.values.copy_((1 - write_strength) * self.values.data + write_strength * updated_values)
        with torch.no_grad():
            self.keys.copy_(F.normalize(self.keys, dim=1))
            self.values.copy_(F.normalize(self.values, dim=1))
```

### Controller

Der Controller besteht aus mehreren Komponenten, darunter ein Embedding-Layer, ein GRU-Layer und mehrere vollständig verbundene Schichten. Der Controller verarbeitet die Eingaben und interagiert mit dem Speicher, um die Ausgaben zu erzeugen.

```python
class Controller(nn.Module):
    def __init__(self, embedding_size: int, memory_embedding_size: int, vocab_size: int, memory_size: int, device: torch.device = DEVICE):
        super(Controller, self).__init__()
        self.memory = DifferentiableMemory(memory_size, memory_embedding_size, device)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc_query = nn.Linear(embedding_size, memory_embedding_size)
        self.fc_query_act = nn.GELU()
        self.rnn = nn.GRU(memory_embedding_size, 256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc1_act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc_output = nn.Linear(256, vocab_size)
        self.fc_output_act = nn.GELU()
        self.fc_query_memory = nn.Linear(memory_embedding_size, memory_embedding_size)
        self.fc_query_memory_act = nn.GELU()
        self.fc_value_memory = nn.Linear(vocab_size, memory_embedding_size)
        self.fc_value_memory_act = nn.GELU()
        self.device = device
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc_query.weight)
        nn.init.zeros_(self.fc_query.bias)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc_output.weight)
        nn.init.zeros_(self.fc_output.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(inputs)
        query = self.fc_query(embedded)
        query = self.fc_query_act(query)
        memory_output = self.memory.read(query)
        rnn_output, _ = self.rnn(memory_output)
        hidden = self.fc1(rnn_output)
        hidden = self.fc1_act(hidden)
        hidden = self.dropout(hidden)
        output = self.fc_output(hidden)
        output = self.fc_output_act(output)
        return output, query
```

## Training

Das Training des Modells erfolgt in mehreren Schritten:

1. **Datenvorbereitung**: Die Daten werden geladen und in Trainings-, Validierungs- und Testdatensätze aufgeteilt.
2. **Modellinitialisierung**: Das Modell und der Optimizer werden initialisiert.
3. **Training-Loop**: Das Modell wird für eine bestimmte Anzahl von Epochen trainiert. Nach jeder Epoche wird das Modell validiert und die Leistung auf dem Testdatensatz evaluiert.
4. **Modellspeicherung**: Das beste Modell wird basierend auf dem Validierungsverlust gespeichert.

```python
def train_model(config: Config):
    tokenizer, vocab = create_vocabulary(config)
    model, optimizer, criterion, scheduler = create_model_and_optimizer(config, len(vocab))
    train_dataloader, val_dataloader, test_dataloader = load_and_prepare_data(config, tokenizer, vocab)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    patience = config.patience
    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []
    test_accuracies = []
    test_f1_scores = []

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, epoch, accumulation_steps=config.accumulation_steps, write_strength=config.write_strength)
        val_loss = validate_model(model, val_dataloader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path / f"memory_model_epoch_{epoch + 1}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                break
        test_loss, accuracy, f1 = evaluate_model(model, test_dataloader, criterion, vocab)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_accuracies.append(accuracy)
        test_f1_scores.append(f1)

    if best_model_state is not None:
        config_to_save = config.save_config()
        torch.save({
            'model_state_dict': best_model_state,
            'vocab': vocab,
            'config': config_to_save
        }, save_path / "memory_model.pth")

    plot_training(train_losses, val_losses, test_accuracies, test_f1_scores)
```

## Evaluierung

Die Evaluierung des Modells erfolgt auf dem Testdatensatz. Die Metriken Accuracy und F1-Score werden verwendet, um die Leistung des Modells zu bewerten.

```python
def evaluate_model(model: nn.Module, test_dataloader: DataLoader, criterion, vocab: Vocabulary) -> Tuple[float, float, float]:
    model.eval()
    test_loss = 0.0
    total_batches = len(test_dataloader)
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for test_inputs, test_targets in tqdm(test_dataloader, desc="Evaluierung"):
            test_inputs, test_targets = test_inputs.to(DEVICE), test_targets.to(DEVICE)
            test_outputs, _ = model(test_inputs)
            test_loss += criterion(F.log_softmax(test_outputs, dim=-1).view(-1, len(model.embedding.weight)), test_targets.view(-1)).item()
            all_predictions.append(test_outputs)
            all_targets.append(test_targets)

    avg_test_loss = test_loss / total_batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy, f1 = calculate_metrics(all_predictions, all_targets, vocab.get_index("<pad>"))
    return avg_test_loss, accuracy, f1
```

## Ergebnisse

Die Ergebnisse des Trainings und der Evaluierung werden in Form von Verlustkurven und Metriken wie Accuracy und F1-Score dargestellt. Diese Ergebnisse zeigen die Leistung des Modells während des Trainings und der Evaluierung.

```python
def plot_training(train_losses, val_losses, test_accuracies, test_f1_scores):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Trainingsverlust')
    plt.plot(val_losses, label='Validierungsverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.title('Trainings- und Validierungsverlust')
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Testgenauigkeit')
    plt.plot(test_f1_scores, label='Test F1-Score')
    plt.xlabel('Epoche')
    plt.ylabel('Metrik')
    plt.legend()
    plt.title('Testgenauigkeit und F1-Score')
    plt.tight_layout()
    plt.show()
```

## Schlussfolgerung

Der Experimentelle DRLMemNet zeigt vielversprechende Ergebnisse bei der Verarbeitung von sequenziellen Daten. Die Integration eines differenzierbaren Speichers und eines Controllers ermöglicht es, Informationen über mehrere Zeitschritte zu speichern und abzurufen, was die Leistung bei Aufgaben wie Textverarbeitung und Sprachmodellierung verbessert.

## Zukünftige Arbeit

Zukünftige Arbeiten könnten sich auf die Erweiterung des Modells konzentrieren, um andere Arten von sequenziellen Daten zu verarbeiten, wie z.B. Zeitreihen oder Audiodaten. Zusätzlich könnten weitere Verbesserungen an der Modellarchitektur und den Trainingsmethoden vorgenommen werden, um die Leistung weiter zu verbessern.

## Literatur

- [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- [2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.


