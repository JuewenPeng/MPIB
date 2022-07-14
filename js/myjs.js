// window.onload = function(){
// var myTab = document.getElementById("paper-method");    //整个div
// var myUl = myTab.getElementsByTagName("ul")[0];//一个节点
// var myLi = myUl.getElementsByTagName("li");    //数组
// var myDiv = myTab.getElementsByTagName("div"); //数组

// for(var i = 0; i<myLi.length;i++){
//     myLi[i].index = i;
//     myLi[i].onclick = function(){
//         for(var j = 0; j < myLi.length; j++){
//             myLi[j].className = "off";
//             myDiv[j].className = "hide";
//         }
//         this.className = "on";
//         myDiv[this.index].className = "show";
//     }
//   }
// }


let methodTabNavItem = document.querySelectorAll("#method .tab_nav>li");
let methodTabBoxItem = document.querySelectorAll("#method .tab_box>li");
let methodTabNavActive = document.querySelector("#method .tabNav_active");
let methodTabBoxActive = document.querySelector("#method .tabBox_active");

methodTabNavItem.forEach(function(item, index) {
    item.addEventListener('click', function() {
        methodTabNavActive.className = "";
        this.className = "tabNav_active";
        methodTabNavActive = this;

        methodTabBoxActive.className = "";
        methodTabBoxItem[index].className = "tabBox_active";
        methodTabBoxActive = methodTabBoxItem[index];
    }, false)
});


let resultTabNavItem = document.querySelectorAll("#result .tab_nav>li");
let resultTabBoxItem = document.querySelectorAll("#result .tab_box>li");
let resultTabNavActive = document.querySelector("#result .tabNav_active");
let resultTabBoxActive = document.querySelector("#result .tabBox_active");

resultTabNavItem.forEach(function(item, index) {
    item.addEventListener('click', function() {
        resultTabNavActive.className = "";
        this.className = "tabNav_active";
        resultTabNavActive = this;

        resultTabBoxActive.className = "";
        resultTabBoxItem[index].className = "tabBox_active";
        resultTabBoxActive = resultTabBoxItem[index];
    }, false)
});

  
// Automatically setup all `image-comparison` elements after the DOM loads
document.addEventListener('DOMContentLoaded', () => {
    const $els = [...document.querySelectorAll('image-comparison')];
    $els.forEach(function makeComparison($el) {
        const fps = 60;
        const throttleDelay = 1000 / fps;

        $el.onmousemove = updatePosition

        function updatePosition(e) {
            const relative = e.offsetX / $el.clientWidth;
            $el.style.setProperty('--current-position', `${relative * 100}%`);
        }
    });
}, false);



var web_left = 'all-in-focus';
var web_right = 'MPIB';
var web_scene = 'new_11'
var web_focus = 'fore'


function getMethodPathWeb(method) {
    var path
    if (method == 'all-in-focus' || method == 'disparity') {
        path = '../img/comparison/' + web_scene + '_' + method + '.jpg'
    }
    else {
        path = '../img/comparison/' + web_scene + '_' + method + '_' + web_focus + '.jpg'
    }
    return path
}



function selectSceneWeb(scene) {
    var select_prev = document.getElementById("select-scene-web-" + web_scene)
    var select = document.getElementById("select-scene-web-" + scene)
    var compare = document.getElementById("compare-method-web")
    var base = document.getElementById("base-method-web")
    var path
    
    select_prev.style.opacity = 0.4
    select.style.opacity = 1
    web_scene = scene

    path = getMethodPathWeb(web_left)
    compare.style.setProperty('--image1', "url(" + path + ")")
    base.src = path.substring(3)

    path = getMethodPathWeb(web_right)
    compare.style.setProperty('--image2', "url(" + path + ")")
}


function selectMethodWeb(side) {
    var select = document.getElementById("select-method-web-" + side)
    var compare = document.getElementById("compare-method-web")
    var path

    var method = select.options[select.selectedIndex].value
    path = getMethodPathWeb(method)
    if (side == 'left') {
        compare.style.setProperty('--image1', "url(" + path + ")")
        web_left = method
    }
    else {
        compare.style.setProperty('--image2', "url(" + path + ")")
        web_right = method
    }
}


function selectFocus(target) {
    web_focus = target
    selectMethodWeb("left")
    selectMethodWeb("right")
}



// var select = document.getElementById("select-method-ebb-277-left");
// select.onselect = function() { //当选项改变时触发
//     var val = this.options[this.selectedIndex].value; //获取option的value
//      // alert(valOption);
//     // var txtOption = this.options[this.selectedIndex].innerHTML; //获取option中间的文本
//      // alert(txtOption);
//     var path = '../img/BLB/277/' + val + '.jpg'
//     var compare = document.getElementById("compare-method-ebb-277")
//     compare.style.setProperty('--image1', "url(" + path + ")")
// }


// toteg(item) {
//     var id = "#" + item;
//     let idItem = document.getElementById(item);
//     let anchor = this.$el.querySelector(id); //计算传进来的id到顶部的距离
//     this.$nextTick(() => {
//         // console.log(anchor.offsetTop)
//         window.scrollTo(0, anchor.offsetTop - 130);  //滚动距离因为导航栏固定定位130px的高度
//     });
// },

// var anchorLink = document.getElementById("icon-link"),
// target = document.getElementById("dataset");
// anchorLink.addEventListener("click", function(e) {
//     if (window.scrollTo) {
//         e.preventDefault();
//         window.scrollTo({"behavior": "smooth", "top": target.offsetTop});
//     }
// })