// ============================================================
// Urban Expansion Monitoring - Indian Metropolitan Areas
// Google Earth Engine Code Editor Script
// ============================================================
// HOW TO USE:
//   1. Go to https://code.earthengine.google.com/
//   2. Paste this entire script
//   3. Click "Run"
//   4. Open the "Tasks" tab (right panel)
//   5. Click "Run" on each export task
//   6. Files will appear in your Google Drive under "urban_expansion_india/"
//
// This exports cloud-free composites for 7 Indian cities across
// 3 time periods (Sentinel-2 2018-2020, 2020-2023, and Landsat 8 2013-2020).
// Each export produces a 6-band GeoTIFF + NDVI + NDBI = 8 bands.
// ============================================================

var cities = {
  'Mumbai':    ee.Geometry.Rectangle([72.75, 18.85, 73.05, 19.30]),
  'Delhi_NCR': ee.Geometry.Rectangle([76.85, 28.40, 77.45, 28.85]),
  'Bangalore': ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.15]),
  'Hyderabad': ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.55]),
  'Chennai':   ee.Geometry.Rectangle([80.15, 12.95, 80.35, 13.20]),
  'Pune':      ee.Geometry.Rectangle([73.75, 18.45, 73.95, 18.65]),
  'Ahmedabad': ee.Geometry.Rectangle([72.50, 22.95, 72.70, 23.15]),
};

// ── Sentinel-2 Cloud Masking ────────────────────────────

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// ── Landsat 8 Cloud Masking (Collection 2 Level 2) ──────

function maskL8clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).eq(0);
  var shadow = qa.bitwiseAnd(1 << 4).eq(0);
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return optical.updateMask(cloud.and(shadow));
}

// ── Landsat 5 Cloud Masking (Collection 2 Level 2) ──────

function maskL5clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).eq(0);
  var shadow = qa.bitwiseAnd(1 << 4).eq(0);
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return optical.updateMask(cloud.and(shadow));
}

// ── Export Helper ────────────────────────────────────────

function exportComposite(cityName, roi, collection, bands, start, end, scale, maskFn, nirIdx, redIdx, swirIdx) {
  var filtered = collection
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskFn);

  var nImages = filtered.size();

  var composite = filtered.select(bands).median().clip(roi);

  // Add spectral indices
  var ndvi = composite.normalizedDifference([bands[nirIdx], bands[redIdx]]).rename('NDVI');
  var ndbi = composite.normalizedDifference([bands[swirIdx], bands[nirIdx]]).rename('NDBI');
  composite = composite.addBands([ndvi, ndbi]);

  var period = start.slice(0,4) + '_' + end.slice(0,4);
  var desc = cityName + '_' + period;

  Export.image.toDrive({
    image: composite,
    description: desc,
    folder: 'urban_expansion_india',
    region: roi,
    scale: scale,
    crs: 'EPSG:4326',
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });

  // Print info
  print(desc + ' — images available: ', nImages);
}

// ── Collections ──────────────────────────────────────────

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));
var s2Bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
// indices: B8=NIR(3), B4=Red(2), B11=SWIR1(4)

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var l8Bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
// indices: SR_B5=NIR(3), SR_B4=Red(2), SR_B6=SWIR1(4)

var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2');
var l5Bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];
// indices: SR_B4=NIR(3), SR_B3=Red(2), SR_B5=SWIR1(4)

var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2');
var l7Bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];

// ── Export All Cities ────────────────────────────────────

print('============================================');
print('Urban Expansion Monitoring - Indian Cities');
print('============================================');

for (var city in cities) {
  var roi = cities[city];

  print('--- ' + city + ' ---');

  // Sentinel-2: 2018-2020 (recent baseline)
  exportComposite(city, roi, s2, s2Bands,
    '2018-01-01', '2020-12-31', 10, maskS2clouds, 3, 2, 4);

  // Sentinel-2: 2020-2023 (most recent)
  exportComposite(city, roi, s2, s2Bands,
    '2020-01-01', '2023-12-31', 10, maskS2clouds, 3, 2, 4);

  // Landsat 8: 2013-2020 (recent historical)
  exportComposite(city, roi, l8, l8Bands,
    '2013-04-01', '2020-12-31', 30, maskL8clouds, 3, 2, 4);

  // Landsat 5: 1990-2000 (historical baseline)
  exportComposite(city, roi, l5, l5Bands,
    '1990-01-01', '2000-12-31', 30, maskL5clouds, 3, 2, 4);

  // Landsat 7: 2000-2012 (mid-period)
  exportComposite(city, roi, l7, l7Bands,
    '2000-01-01', '2012-12-31', 30, maskL5clouds, 3, 2, 4);
}

print('============================================');
print('All export tasks created.');
print('Open the Tasks tab and click Run on each.');
print('Files will appear in Google Drive: urban_expansion_india/');
print('============================================');

// ── Visualization (optional) ─────────────────────────────
// Uncomment below to visualize Mumbai as an example

// var mumbaiS2 = s2.filterBounds(cities['Mumbai'])
//     .filterDate('2020-01-01', '2023-12-31')
//     .map(maskS2clouds)
//     .select(s2Bands)
//     .median()
//     .clip(cities['Mumbai']);
//
// Map.centerObject(cities['Mumbai'], 11);
// Map.addLayer(mumbaiS2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'Mumbai RGB');
// Map.addLayer(mumbaiS2.normalizedDifference(['B8', 'B4']),
//     {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'Mumbai NDVI');
