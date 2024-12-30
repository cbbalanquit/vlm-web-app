from flask import Blueprint, render_template, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/health')
def health():
    return jsonify({
        'status': 'success',
        'message': 'API is healthy'
    }), 200