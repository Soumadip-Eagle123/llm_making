import graphviz
from flask import Flask, Response, jsonify, request, send_from_directory
from engine import Session, parse_config

app = Flask(__name__, static_folder='static', static_url_path='/static')
S = {'s': None}


@app.errorhandler(ValueError)
@app.errorhandler(OverflowError)
def bad(e):
    msg = str(e) if isinstance(e, ValueError) else 'Numeric overflow: lower the learning rate and rebuild.'
    return jsonify(error=msg), 400


def need():
    if S['s'] is None:
        raise ValueError('Build a model first.')
    return S['s']


@app.get('/')
def index():
    return send_from_directory('static', 'index.html')


@app.post('/api/init')
def init():
    S['s'] = Session(parse_config(request.get_json(force=True)))
    return jsonify(params=S['s'].initial, nparams=len(S['s'].params))


@app.post('/api/iteration')
def iteration():
    return jsonify(need().iteration())


@app.post('/api/update')
def update():
    s = need()
    ev = s.update()
    return jsonify(events=ev, params={p.label: p.data for p in s.params})


@app.get('/api/export.jpg')
def export():
    s = need()
    if not s.dot_src:
        raise ValueError('Nothing to export yet.')
    try:
        data = graphviz.Source(s.dot_src, format='jpg').pipe()
    except graphviz.ExecutableNotFound:
        return jsonify(error='Graphviz not installed: run `sudo apt install graphviz`.'), 500
    except graphviz.CalledProcessError as e:
        return jsonify(error=f'Graphviz failed: {e}'), 500
    if data[:2] != b'\xff\xd8':
        return jsonify(error='Graphviz did not produce a valid JPEG.'), 500
    return Response(data, mimetype='image/jpeg',
                    headers={'Content-Disposition': 'attachment; filename=computational_graph.jpg'})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)