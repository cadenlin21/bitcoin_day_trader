from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from main import load_data, simulate, run_simulation, generate_plots  # Import the necessary functions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'  # Define the static folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.secret_key = 'your_secret_key'  # Needed for session
# server = app.server
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        use_default = 'use_default' in request.form
        if use_default:
            # Paths to your default files
            bitcoin_filepath = os.path.join(app.root_path, 'static/default', 'BCHAIN-MKPRU.csv')
            gold_filepath = os.path.join(app.root_path, 'static/default', 'LBMA-GOLD.csv')
        else:
            bitcoin_file = request.files.get('bitcoin_file')
            gold_file = request.files.get('gold_file')

            if not bitcoin_file or not gold_file:
                flash('Both Bitcoin and Gold data files are required.')
                return redirect(request.url)

            if bitcoin_file and allowed_file(bitcoin_file.filename) and gold_file and allowed_file(gold_file.filename):
                bitcoin_filename = secure_filename(bitcoin_file.filename)
                gold_filename = secure_filename(gold_file.filename)
                bitcoin_filepath = os.path.join(app.config['UPLOAD_FOLDER'], bitcoin_filename)
                gold_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gold_filename)
                bitcoin_file.save(bitcoin_filepath)
                gold_file.save(gold_filepath)
            else:
                flash('Invalid file type. Only CSV files are allowed.')
                return redirect(request.url)

            # Run the simulation and generate plots
        results = load_data(bitcoin_filepath, gold_filepath)
        simulation_results = simulate(results)
        static_dir = os.path.join(app.root_path, app.config['STATIC_FOLDER'])
        plot_filenames = generate_plots(results, static_dir)

        # Store results and filenames in session
        session['simulation_results'] = simulation_results
        session['plot_filenames'] = plot_filenames

        return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    # Retrieve results and filenames from session
    simulation_results = session.get('simulation_results', [])
    plot_filenames = session.get('plot_filenames', [])

    return render_template('results.html',
                           simulation_results=simulation_results,
                           plot_filenames=plot_filenames)

if __name__ == '__main__':
    app.run(debug=True)
