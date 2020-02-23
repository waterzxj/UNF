#!usr/bin/env python  
# -*- coding: utf-8 -*-  
from flask import Flask
from server import debug
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

#app = Flask(__name__, static_url_path='/sim/static')
app = Flask(__name__)

# Register blueprint
app.register_blueprint(debug, url_prefix='/debug')



if __name__ == '__main__':

    print('start run_app...')
    # asynchronous load data
    # load_data()
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(4444)
    IOLoop.instance().start()
    # port = 4444
    # app.run(host='0.0.0.0', port=port, debug=True)
