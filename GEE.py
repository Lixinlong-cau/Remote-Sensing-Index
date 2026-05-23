// ===================== 基础配置 =====================
// 加载23个点位资产（直接读取自带name列）
var points = ee.FeatureCollection("projects/ee-cassyb54598/assets/2014201509");
var start = '2014-01-01';
var end = '2019-12-31';
var bufferSize = 500; // 每个点独立500m处理范围

// ===================== 单点位独立处理 =====================
var processSinglePoint = function(feat) {
  var pointGeom = feat.geometry();
  var pointName = feat.get('name'); 
  var pointBuffer = pointGeom.buffer(bufferSize); 

  // ==============================================
  // 1. Landsat8
  // ==============================================
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(pointBuffer)
    .filterDate(start, end)
    .select(['SR_B3','SR_B4','SR_B5','SR_B6','QA_PIXEL'])
    .map(function(img){
      var qa = img.select('QA_PIXEL');
      // 严格去云：清除 云、云阴影、卷云、饱和像素
      var mask = qa.bitwiseAnd(1 << 3).eq(0)  // 无云
                  .and(qa.bitwiseAnd(1 << 4).eq(0))  // 无云阴影
                  .and(qa.bitwiseAnd(1 << 1).eq(0)) // 无卷云
                  .and(qa.bitwiseAnd(1 << 0).eq(0)); // 无饱和
      var sr = img.select(['SR_B3','SR_B4','SR_B5','SR_B6'])
                  .multiply(0.0000275).add(-0.2).updateMask(mask);
      return sr.select(['SR_B3','SR_B4','SR_B5','SR_B6'],['Green','Red','NIR','SWIR'])
               .set('system:time_start',img.get('system:time_start'),'source','L8');
    });

  // ==============================================
  // 2. MODIS
  // ==============================================
  var modis = ee.ImageCollection('MODIS/061/MOD09GA')
    .merge(ee.ImageCollection('MODIS/061/MYD09GA'))
    .filterBounds(pointBuffer)
    .filterDate(start, end)
    .select(['sur_refl_b01','sur_refl_b02','sur_refl_b04','sur_refl_b06','QC_500m'])
    .map(function(img){
      var qc = img.select('QC_500m');
      // 严格去云：云、云阴影、气溶胶污染全部屏蔽
      var mask = qc.bitwiseAnd(1 << 2).eq(0)  // 无云
                  .and(qc.bitwiseAnd(1 << 3).eq(0))  // 无云阴影
                  .and(qc.bitwiseAnd(1 << 4).eq(0)); // 高气溶胶/污染去除
      var sr = img.select(['sur_refl_b01','sur_refl_b02','sur_refl_b04','sur_refl_b06'])
                  .multiply(0.0001).updateMask(mask);
      return sr.select(['sur_refl_b01','sur_refl_b02','sur_refl_b04','sur_refl_b06'],['Red','NIR','Green','SWIR'])
               .set('system:time_start',img.get('system:time_start'),'source','MODIS');
    });

  // ==============================================
  // 3. Sentinel-2
  // ==============================================
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(pointBuffer)
    .filterDate('2015-06-23', end)
    .select(['B3','B4','B8','B11','QA60'])
    .map(function(img){
      var qa = img.select('QA60');
      // 严格去云：云 + 卷云 + 阴影
      var mask = qa.bitwiseAnd(1 << 10).eq(0)  // 无云
                  .and(qa.bitwiseAnd(1 << 11).eq(0)); // 无卷云
      var sr = img.select(['B3','B4','B8','B11'])
                  .divide(10000).updateMask(mask);
      return sr.select(['B3','B4','B8','B11'],['Green','Red','NIR','SWIR'])
               .set('system:time_start',img.get('system:time_start'),'source','S2');
    });

  // ==============================================
  // 统一提取
  // ==============================================
  var merged = l8.merge(modis).merge(s2).sort('system:time_start');
  var extract = function(image) {
    var val = image.reduceRegion({
      reducer: ee.Reducer.first(), geometry: pointGeom, scale: 30, maxPixels: 1e9
    });
    return ee.Feature(null,{
      'name': pointName,
      'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
      'source': image.get('source'),
      'Green': val.get('Green'), 'Red': val.get('Red'),
      'NIR': val.get('NIR'), 'SWIR': val.get('SWIR')
    });
  };
  return merged.map(extract);
};

// ===================== 执行+导出 =====================
var result = points.map(processSinglePoint).flatten();
var finalResult = result.filter(ee.Filter.notNull(['Green','Red','NIR','SWIR']));

Export.table.toDrive({
  collection: finalResult,
  description: '23Points_OriginalName_Bands_ClearMask',
  folder: 'TimeSeries_KF',
  fileFormat: 'CSV',
  selectors: ['name','date','source','Green','Red','NIR','SWIR']
});