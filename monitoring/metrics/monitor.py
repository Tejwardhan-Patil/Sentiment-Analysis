import os
import json
import time
import psutil
import logging
import smtplib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email configuration for alerts
SMTP_SERVER = 'smtp.mailserver.com'
SMTP_PORT = 587
SENDER_EMAIL = 'alert@website.com'
SENDER_PASSWORD = 'password'
RECIPIENT_EMAIL = 'admin@website.com'

# Setup logger
logging.basicConfig(filename='monitor.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to store metrics
metrics_dir = "metrics"
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

# Directory to store graphs
graphs_dir = "graphs"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

# Function to send alert email on performance degradation
def send_alert_email(subject, body, attachment=None):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if attachment:
        filename = os.path.basename(attachment)
        attachment = open(attachment, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {filename}')
        msg.attach(part)
    
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    text = msg.as_string()
    server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
    server.quit()
    logging.info(f"Alert email sent: {subject}")

# Function to log system stats
def log_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    logging.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_info.percent}%, "
                 f"Disk Usage: {disk_usage.percent}%, Network: Sent={net_io.bytes_sent} bytes, "
                 f"Received={net_io.bytes_recv} bytes")

# Function to log metrics
def log_metrics(y_true, y_pred, epoch, loss):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_file = os.path.join(metrics_dir, f'epoch_{epoch}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics for epoch {epoch}: {metrics}")
    
    # Trigger alert if performance drops
    if metrics['accuracy'] < 0.70:  # Threshold
        send_alert_email(f'Performance Alert: Epoch {epoch}', 
                         f"Model accuracy dropped below 70%. Current accuracy: {metrics['accuracy']}",
                         metrics_file)

# Function to plot and save graphs
def plot_metrics_graph(epochs, metric_values, metric_name):
    plt.figure()
    plt.plot(epochs, metric_values, marker='o', label=metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    graph_file = os.path.join(graphs_dir, f'{metric_name}_graph.png')
    plt.savefig(graph_file)
    logging.info(f"Graph saved: {graph_file}")
    plt.close()

# Function to monitor and log model performance during training
def monitor_performance(model, data_loader, criterion, epoch, metrics_tracker):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    import torch
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_loss = total_loss / len(data_loader)
    metrics_tracker['accuracy'].append(accuracy_score(y_true, y_pred))
    metrics_tracker['f1_score'].append(f1_score(y_true, y_pred, average='weighted'))
    metrics_tracker['loss'].append(epoch_loss)

    log_metrics(y_true, y_pred, epoch, epoch_loss)
    log_system_stats()

# Function to generate real-time metrics graphs
def generate_metrics_graphs(metrics_tracker):
    epochs = list(range(1, len(metrics_tracker['accuracy']) + 1))
    plot_metrics_graph(epochs, metrics_tracker['accuracy'], 'accuracy')
    plot_metrics_graph(epochs, metrics_tracker['f1_score'], 'f1_score')
    plot_metrics_graph(epochs, metrics_tracker['loss'], 'loss')

# Function to aggregate metrics and send report via email
def send_metrics_report():
    metrics_files = [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith('.json')]
    metrics_summary = []
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics_summary.append(json.load(f))
    
    summary_file = os.path.join(metrics_dir, 'metrics_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    send_alert_email('Metrics Summary Report', 'Attached is the summary of model metrics.', summary_file)

# Usage within a training loop
# metrics_tracker = {'accuracy': [], 'f1_score': [], 'loss': []}
# for epoch in range(1, num_epochs+1):
#     monitor_performance(model, val_loader, criterion, epoch, metrics_tracker)
#     generate_metrics_graphs(metrics_tracker)
# send_metrics_report()