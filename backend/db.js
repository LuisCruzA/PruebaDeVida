const { Pool } = require('pg');

const pool = new Pool({
  connectionString: 'postgresql://postgres:BkrenHJsnojokkpMRVxteThvUDQorufz@mainline.proxy.rlwy.net:46301/railway',
  ssl: {
    rejectUnauthorized: false
  }
});

async function insertarUsuario({ nombre, apellido, correo, contraseña }) {
  const res = await pool.query(
    'INSERT INTO users (nombre, apellido, correo, contraseña) VALUES ($1, $2, $3, $4) RETURNING *;',
    [nombre, apellido, correo, contraseña]
  );
  return res.rows[0];
}

async function obtenerUsuarios() {
  const res = await pool.query('SELECT * FROM users;');
  return res.rows;
}

module.exports = {
  insertarUsuario,
  obtenerUsuarios
};
