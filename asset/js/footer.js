document.addEventListener('DOMContentLoaded', function () {
    var normal = document.getElementById('footer');
    if (normal) {
        var content = `
            <div class="container text-center py-3 small border-top" style="margin-top: 100px;">
                <p class="text-center" style="margin-top: 50px;"> &copy; GarmentIQ.ly.gd.edu.kg </p>
                <p class="text-center" xmlns:cc="http://creativecommons.org/ns#"> Unless otherwise specified, this webpage is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" class="link-fancy" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution 4.0 International <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="">
                        <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="">
                    </a>.
                </p>
                <p class="text-center">Last update of this website: 04/21/2025</p>
                <p class="text-center">
                    <a class="link-fancy" target="_blank" href="https://www.gd.edu.kg/privacy-policy/">Privacy Policy</a>
                    <a class="link-fancy" target="_blank" href="https://www.gd.edu.kg/cookie-policy/">Cookie Policy</a>
                    <a class="link-fancy" target="_blank" href="https://garmentiq.ly.gd.edu.kg/sitemap.xml">Site Map</a>
                </p>
                <p class="text-center" style="margin-top: 50px; font-size: 8px;"> Website template provided by <a href="https://templatedeck.com" class="link-fancy" target="_blank">templatedeck.com</a>. </p>
            </div>
        `;
        normal.innerHTML = content;
    } else {
        console.error('Container not found. Make sure the div with id "normalMenu" exists.');
    }
});
