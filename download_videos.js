const https = require('https');
const fs = require('fs');
const url = 'https://trafficcams.bellevuewa.gov/traffic-edge/CCTV074L.stream/playlist.m3u8';

https.get(url, (response) => {
    let data = '';
    response.on('data', (chunk) => {
	data+=chunk;
    });
    response.on('end' , () => {
	const search_for_m3u8 = /.*\.m3u8/g;
	const chunklist_Url = data.match(search_for_m3u8);

	if (chunklist_Url) {
	    console.log(chunklist_Url);
	    get_chunklist_ts_files(chunklist_Url);
	} else {
	    console.log('No m3u8 files detected');
	}
    });
});

function get_chunklist_ts_files(url2) {
    const url3 = `https://trafficcams.bellevuewa.gov/traffic-edge/CCTV074L.stream/${url2}`;
    https.get(url3, (response) => {
	let data = '';
	response.on('data', (chunk) => {
	    data+=chunk;
	});

	response.on('end', () => {
	    const tsFileRegex = /.*\.ts/g;
	    const tsFileUrls = data.match(tsFileRegex);

	    if (tsFileUrls) {
		download_ts_files(tsFileUrls[0]);
	    } else {
		console.log('Error parsing playlist.m3u8');
	    }
	});
    });
}

function download_ts_files(url4){
    const url5 = `https://trafficcams.bellevuewa.gov/traffic-edge/CCTV074L.stream/${url4}`;
    const file = fs.createWriteStream('traffic_cam_videos/traffic_cam.ts');
    https.get(url5, (response) => {
	response.pipe(file);

	file.on('finish', () => {
	    file.close();
	    console.log('Download complete');
	});
    }).on('error', (err) => {
	fs.unlink('traffic_cam_videos/traffic_cam.ts');
	console.error('Download failed:', err.message);
    });
}
