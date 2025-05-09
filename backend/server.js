// backend/server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('express').json;
const { insertarUsuario, obtenerUsuarios } = require('./db');

const app = express();
app.use(cors());
app.use(bodyParser());

app.post('/api/usuarios', async (req, res) => {
  try {
    const nuevoUsuario = await insertarUsuario(req.body);
    res.json(nuevoUsuario);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al insertar usuario' });
  }
});

app.get('/api/usuarios', async (req, res) => {
  try {
    const usuarios = await obtenerUsuarios();
    res.json(usuarios);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error al obtener usuarios' });
  }
});


app.listen(3000, () => {
  console.log('Servidor corriendo en http://localhost:3000');
});
