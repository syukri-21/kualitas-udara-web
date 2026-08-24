// -----------------------------------------------------------------------
// Logika frontend: ambil input form -> POST ke /predict -> render hasil
// -----------------------------------------------------------------------
const form = document.getElementById('airForm');
const btn = document.getElementById('btnCheck');
const formError = document.getElementById('formError');

const resultCard = document.getElementById('resultCard');
const resultEmpty = document.getElementById('resultEmpty');
const resultContent = document.getElementById('resultContent');

const resKategori = document.getElementById('resKategori');
const resConfidence = document.getElementById('resConfidence');
const resDeskripsi = document.getElementById('resDeskripsi');
const resSaran = document.getElementById('resSaran');
const horizonMarker = document.getElementById('horizonMarker');

// Posisi marker di sepanjang horizon bar (dalam %) per kategori.
// Tiap zona lebar 1/3, marker diletakkan di tengah zona kategori terkait.
const MARKER_POSITION = {
  'BAIK': 16.6,
  'SEDANG': 50,
  'TIDAK SEHAT': 83.3,
};

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  formError.textContent = '';

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value;
  }

  setLoading(true);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok) {
      formError.textContent = data.error || 'Terjadi kesalahan, coba lagi.';
      return;
    }

    renderResult(data);
  } catch (err) {
    formError.textContent = 'Tidak bisa menghubungi server. Pastikan backend berjalan.';
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  btn.disabled = isLoading;
  btn.classList.toggle('is-loading', isLoading);
}

function renderResult(data) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;
  resultCard.setAttribute('data-kategori', data.kategori);

  resKategori.textContent = data.kategori;
  resConfidence.textContent = `Tingkat keyakinan model: ${data.confidence}%`;
  resDeskripsi.textContent = data.deskripsi;

  resSaran.innerHTML = '';
  data.saran.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    resSaran.appendChild(li);
  });

  const pos = MARKER_POSITION[data.kategori];
  if (pos !== undefined) {
    horizonMarker.style.left = `${pos}%`;
  }

  resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
