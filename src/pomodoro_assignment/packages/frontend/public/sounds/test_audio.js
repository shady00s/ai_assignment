// Simple test script to verify audio setup
// This can be run in browser console to test the ambient sound functionality

console.log('🎵 Ambient Sound Test Script');
console.log('=====================================');

// Test audio context availability
if (typeof window !== 'undefined' && window.AudioContext) {
  console.log('✅ Web Audio API is available');
} else {
  console.log('❌ Web Audio API is not available');
}

// Test Howler.js availability
if (typeof window !== 'undefined' && window.Howl) {
  console.log('✅ Howler.js is available');
} else {
  console.log('❌ Howler.js is not available');
}

// Test sound files
const soundFiles = ['forest.mp3', 'ocean.mp3', 'cafe.mp3', 'rain.mp3'];
soundFiles.forEach(file => {
  fetch(`/sounds/${file}`)
    .then(response => {
      if (response.ok) {
        console.log(`✅ ${file} - Found`);
      } else {
        console.log(`❌ ${file} - Not found (${response.status})`);
      }
    })
    .catch(error => {
      console.log(`❌ ${file} - Error: ${error.message}`);
    });
});

console.log('Test ambient sound by toggling the Ambient Settings in the Timer Screen!');