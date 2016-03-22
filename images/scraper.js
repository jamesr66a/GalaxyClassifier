var list = document.getElementsByClassName('image-list')
Array.prototype.slice.call(list[0].childNodes).forEach(function (obj) {
  console.log(obj.childNodes[0].getAttribute('src'))
})
