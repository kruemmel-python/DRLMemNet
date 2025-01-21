# DRLMemNet: Differentiable Recurrent Learning with Memory Network

## Modellbeschreibung

Das `DRLMemNet`-Modell ist eine auf neuronalen Netzen basierende Architektur für die Verarbeitung von sequentiellen Daten, insbesondere für Aufgaben wie die Textgenerierung. Es kombiniert ein rekurrentes neuronales Netzwerk (RNN) mit einem differenzierbaren Speicher (Memory Network). Das Modell zielt darauf ab, durch die Nutzung eines expliziten Speichers die Fähigkeit zur Verarbeitung von längeren Sequenzen und komplexeren Beziehungen zwischen Elementen zu verbessern.

### Kernkomponenten

1.  **DifferentiableMemory**:
    *   Implementiert einen externen, differenzierbaren Speicher, der aus Key- und Value-Matrizen besteht.
    *   Die Speicherung erfolgt durch das Normalisieren der Keys und Values.
    *   Die Lesefunktion verwendet ein Softmax über die Ähnlichkeit zwischen einer Query und den Keys, um die Memory-Inhalte zu gewichten.
    *   Die Schreibfunktion nutzt einen gewichteten Mittelwert aus bestehenden und neuen Inhalten.
    *   Verwaltet Nutzungscounts für jede Memory-Zelle.
2.  **Controller**:
    *   Das Hauptmodell, welches die Memory-Komponente nutzt.
    *   Besteht aus einem Embedding-Layer zur Umwandlung von Token-Indizes in dichte Vektoren.
    *   Verwendet eine lineare Schicht, um die Embedding-Vektoren in Query-Vektoren für das Memory zu transformieren.
    *   Nutzt ein GRU (Gated Recurrent Unit) als rekurrentes Netzwerk zur Verarbeitung der aus dem Memory gelesenen Informationen.
    *   Enthält lineare Schichten mit GELU-Aktivierungsfunktionen und Dropout für die finale Vorhersage.
    *   Integrierte lineare Schichten für Memory-Updates, die durch die mittlere Embedding-Vektoren für Keys und Values aufgerufen werden.
    *   Initialisiert die Gewichte der linearen Layer mit der Kaiming-Uniform-Methode und die Embedding-Layer mit einer Normalverteilung.
3.  **Vocabulary**:
    *   Verwaltet die Zuordnung von Token zu Indizes und umgekehrt.
    *   Beinhaltet spezielle Tokens wie `<pad>`, `<unk>`, `<sos>` (Start of Sequence) und `<eos>` (End of Sequence).
    *   Bietet Methoden zum Hinzufügen von Tokens und zum Abrufen von Indizes/Tokens.
4.  **TextDataset**:
    *   Liest Zeilen aus einer Textdatei und konvertiert sie in indizierte Eingabe- und Zielsequenzen.
    *   Tokenisiert Texte und wandelt sie in numerische Sequenzen um.
    *   Erzeugt Trainingsdaten mit Padding, um eine einheitliche Sequenzlänge zu gewährleisten.

### Trainingsprozess

1.  **Konfiguration**:
    *   Verwendet eine `Config`-Klasse zum Laden und Validieren von Konfigurationsparametern.
    *   Die Konfiguration kann entweder aus einer YAML-Datei oder einem Dictionary (z.B. der Ergebnis des Optuna-Tunings) geladen werden.
    *   Enthält Parameter wie:
        *   `data_path`: Pfad zur Eingabedatei
        *   `log_file`: Pfad zur Logdatei
        *   `embedding_dim`: Dimension des Embeddings
        *   `memory_size`: Größe des Memory
        *   `learning_rate`: Lernrate des Optimierers
        *   `batch_size`: Anzahl der Elemente pro Batch
        *   `max_seq_length`: Maximale Sequenzlänge
        *   `train_size_ratio`, `val_size_ratio`: Aufteilungsverhältnis für Trainings- und Validierungsdaten
        *   `epochs`: Anzahl der Trainingsepochen
        *   `accumulation_steps`: Anzahl der Schritte zur Gradientenakkumulation
        *   `write_strength`: Schreibstärke für das Memory
        *   `patience`: Geduld für Early Stopping
        *   `save_path`: Pfad zum Speichern des Modells
        *   `device`: Gerät (CPU oder GPU)
2.  **Datenvorbereitung**:
    *   Erstellt ein Vokabular basierend auf den Tokens der Eingabedaten.
    *   Initialisiert ein `TextDataset`, um die Trainingsdaten vorzubereiten.
    *   Erstellt DataLoaders für Training, Validierung und Test.
3.  **Modell und Optimierer**:
    *   Initialisiert das `Controller`-Modell mit den konfigurierten Parametern.
    *   Verwendet den AdamW-Optimierer mit einer Cosine Annealing Learning Rate Scheduler.
    *   Verwendet die Cross-Entropy-Loss-Funktion als Kriterium.
4.  **Training**:
    *   Trainiert das Modell über mehrere Epochen.
    *   Verwendet optional Gradientenakkumulation.
    *   Führt eine Validierung nach jeder Epoche durch.
    *   Führt Early Stopping durch, falls der Validierungsverlust sich nicht verbessert.
    *   Speichert das beste Modell basierend auf dem Validierungsverlust.
    *   Memory Update wird nach dem Gradientenupdate durchgeführt.
5.  **Evaluierung**:
    *   Berechnet Loss, Genauigkeit und F1-Score auf dem Test-Dataset.
    *   Verwendet den `calculate_metrics`-Funktion um die Metriken auf den gemaskten Daten zu berechnen.
    *   Der Evaluation wird speicherschonend batchweise ausgeführt.
6.  **Optuna-Integration**:
    *   Verwendet Optuna zur automatischen Hyperparameteroptimierung.
    *   Das `objective`-Funktion definiert die zu optimierenden Parameter.
    *   Die besten Hyperparameter werden zur finalen Trainingsrunde verwendet.
7.  **Visualisierung**:
    *   Plotet den Trainings- und Validierungsverlust sowie die Testgenauigkeit und den F1-Score.
    *   Sichert die generierten Plots ab.
8.  **Speicherung**:
    *   Speichert das beste Modell (Gewichte), das Vokabular und die Konfiguration.
    *   Die gespeicherten Modelle können geladen und zur Vorhersage verwendet werden.

### Implementierung

*   Die Implementierung ist in Python geschrieben und verwendet `PyTorch` für die Modellierung und das Training.
*   Zusätzliche Bibliotheken sind `optuna` (für Hyperparameter-Tuning), `tqdm` (für Fortschrittsanzeigen), `matplotlib` (für Visualisierungen) und `PyYAML` (für Konfigurationsverwaltung).
*   Die Logging-Funktionalität ist über das `logging`-Modul implementiert.
*   Die Verwendung von Data Classes, um Parameter zu übergeben und zu verwalten, unterstützt die Codeklarheit.

### Hauptfunktionen

*   `train_model()`: Hauptfunktion, die den Trainingsprozess steuert.
*   `objective()`: Funktion für die Optuna-Hyperparameteroptimierung.
*   `train_model_epoch()`: Führt eine einzelne Trainings-Epoche durch.
*   `validate_model()`: Validiert das Modell auf dem Validierungs-Dataset.
*   `evaluate_model()`: Evaluiert das Modell auf dem Test-Dataset.
*   `create_data_loaders()`: Erstellt DataLoaders für Training, Validierung und Test.
*   `load_and_prepare_data()`: Lädt und bereitet Daten vor.
*   `create_model_and_optimizer()`: Erstellt das Modell, den Optimierer, das Kriterium und den Learning Rate Scheduler.
*   `create_vocabulary()`: Erstellt das Vokabular.
*   `plot_training()`: Visualisiert die Trainingsmetriken.
*   `setup_training()`: Setzt die Trainingseinstellungen (Modell, Optimierer, Kriterium, Scheduler, DataLoaders).
*   `run_training_loop()`: Führt die Trainingsschleife aus.
*   `save_best_model()`: Speichert das beste Modell, das Vokabular und die Konfiguration.

### Anwendung

Das `DRLMemNet`-Modell kann für eine Vielzahl von sequenziellen Datenverarbeitungsaufgaben verwendet werden, wie z. B.:

*   Textgenerierung
*   Maschinelle Übersetzung
*   Textzusammenfassung
*   Sequenzklassifizierung
*   Dialogsysteme

## Installation

1.  Stellen Sie sicher, dass Python (Version 3.8 oder höher) und pip installiert sind.
2.  Installieren Sie die erforderlichen Bibliotheken mit dem Befehl:

    ```bash
    pip install torch optuna tqdm PyYAML scikit-learn matplotlib
    ```

## Konfiguration

1.  Erstellen Sie eine Datei `config.yaml` im Stammverzeichnis Ihres Projektes mit folgenden Inhalten:

    ```yaml
    data_path: "data.txt"
    log_file: "training.log"
    embedding_dim: 256
    memory_size: 1024
    learning_rate: 0.001
    batch_size: 32
    max_seq_length: 50
    train_size_ratio: 0.8
    val_size_ratio: 0.1
    epochs: 100
    accumulation_steps: 2
    write_strength: 0.05
    patience: 10
    save_path: "model_checkpoints"
    ```

    *   Passen Sie die Werte entsprechend Ihren Bedürfnissen an.

## Ausführung

1.  Speichern Sie die obige Implementierung in einer Datei z.B. `drlmemnet.py`.
2.  Führen Sie das Training aus, indem Sie folgendes Kommando in der Konsole ausführen:

    ```bash
    python drlmemnet.py
    ```

    *   Das Training startet automatisch und die Ergebnisse werden in der Kommandozeile und im `training.log` angezeigt.
    *   Das beste trainierte Modell wird im angegebenen Pfad gespeichert (defaut: `"model_checkpoints"`)

## Weiterführende Informationen

*   Der Code ist mit ausführlichen Kommentaren versehen, um die einzelnen Schritte zu erläutern.
*   Die Hyperparameteroptimierung mit Optuna kann durch Anpassen der Optuna-Parameter in der Funktion `train_model` weiter angepasst werden.
*   Der verwendete Datensatz (`data.txt`) muss pro Zeile eine Trainingssequenz enthalten.
*   Die Logging-Ausgabe hilft bei der Beobachtung des Trainingsfortschritts und bei der Fehlersuche.


