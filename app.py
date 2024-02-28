# source env/bin/activate
# python app.py

from flask import Flask, render_template, request
import plots as plots

app = Flask(__name__)

g00 = None
f0 = None
f1 = None
d = None
Ms = None
Bs = None
lam_exp = None
noise = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulation', methods=['POST'])
def plot():
    global g00, f0, f1, d, Ms, Bs, lam_exp, noise, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10

    # Get user input from the form
    _g00 = int(request.form['g00'])
    _f0 = int(request.form['f0'])
    _f1 = int(request.form['f1'])
    _d = int(request.form['d'])
    _Ms = int(request.form['Ms'])
    _Bs = float(request.form['Bs'])
    _lam_exp = float(request.form['lam_exp'])
    _noise = float(request.form['noise'])

    if g00 == _g00 and f0 == _f0 and f1 == _f1 and d == _d and Ms == _Ms and Bs == _Bs and noise == _noise:
        if lam_exp == _lam_exp:
            print("NO CHANGE")
        else:
            print("LAM CHANGE")
            lam_exp = _lam_exp
            d8, d9, d10 = plots.lambda_dependent_graphs(g00, f0, f1, d, Ms, Bs, lam_exp, noise)
    else:
        print("SOMETHING CHANGE")
        g00, f0, f1, d, Ms, Bs, lam_exp, noise = _g00, _f0, _f1, _d, _Ms, _Bs, _lam_exp, _noise
        d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = plots.signal_generation(g00, f0, f1, d, Ms, Bs, lam_exp, noise)

    return render_template('simulation_graph.html', data1=d1, data2=d2, data3=d3, data4=d4, data5=d5, data6=d6, data7=d7, data8=d8, data9=d9, data10=d10)

@app.route('/simulation', methods=['GET'])
def simulation():
    return render_template('simulation.html')    

if __name__ == "__main__":
    app.run(debug=True)